import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import json
from torch.utils.data import Dataset, DataLoader


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device

# Import your modules and utils
from env import PositionBreakEnv, MaskedMultiBinaryPolicy
from models import GAILDiscriminator
from utils import (
    load_expert_data_from_json,
    preprocess_obs,
    GAILRewardWrapper
)


def load_multiple_expert_data(file_paths):
    """
    Load and concatenate expert data from multiple JSON files.
    Assuming each JSON file has structure:
    {
      "expert_obs": {"ms_trades": ..., "exchange_trades": ..., "positions": ..., "position_diff": ...},
      "expert_actions": [...]
    }
    Returns combined expert_obs (dict of arrays) and expert_actions (array).
    """
    all_ms_trades = []
    all_exchange_trades = []
    all_positions = []
    all_position_diff = []
    all_actions = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        obs = data["expert_obs"]
        actions = data["expert_actions"]

        ms_trades = np.array(obs['ms_trades'], dtype=np.float32)
        exchange_trades = np.array(obs['exchange_trades'], dtype=np.float32)
        positions = np.array(obs['positions'], dtype=np.float32)
        position_diff = np.array(obs['position_diff'], dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        all_ms_trades.append(ms_trades)
        all_exchange_trades.append(exchange_trades)
        all_positions.append(positions)
        all_position_diff.append(position_diff)
        all_actions.append(actions)

    all_ms_trades = np.concatenate(all_ms_trades, axis=0)
    all_exchange_trades = np.concatenate(all_exchange_trades, axis=0)
    all_positions = np.concatenate(all_positions, axis=0)
    all_position_diff = np.concatenate(all_position_diff, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    expert_obs = {
        'ms_trades': all_ms_trades,
        'exchange_trades': all_exchange_trades,
        'positions': all_positions,
        'position_diff': all_position_diff
    }

    return expert_obs, all_actions


######################################
# BC dataset and training
######################################

class ExpertDataset(Dataset):
    def __init__(self, expert_obs, expert_actions):
        self.expert_obs = expert_obs
        self.expert_actions = expert_actions
        self.num_samples = expert_actions.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs_sample = {k: self.expert_obs[k][idx] for k in self.expert_obs}
        action_sample = self.expert_actions[idx]
        return obs_sample, action_sample

def collate_fn(batch):
    obs_list = [item[0] for item in batch]
    actions_list = [item[1] for item in batch]

    actions_tensor = torch.tensor(actions_list, dtype=torch.float32)
    batch_obs = {}
    for key in obs_list[0].keys():
        batch_obs[key] = torch.tensor([obs[key] for obs in obs_list], dtype=torch.float32)

    return batch_obs, actions_tensor

def behavior_cloning_pretrain(policy, expert_obs, expert_actions, device, epochs=10, batch_size=64, lr=1e-3):
    policy.to(device)
    policy.train()

    dataset = ExpertDataset(expert_obs, expert_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = Adam(policy.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in dataloader:
            batch_actions = batch_actions.to(device)
            for k in batch_obs:
                batch_obs[k] = batch_obs[k].to(device)

            # Forward pass through the policy
            features = policy.extract_features(batch_obs)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            probabilities = torch.sigmoid(logits)  # multi-binary actions

            loss = criterion(probabilities, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"BC Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    policy.eval()
    return policy

def extract_initial_observation_from_expert(expert_obs):
    init_obs = {
        'ms_trades': expert_obs['ms_trades'][0].copy(),
        'exchange_trades': expert_obs['exchange_trades'][0].copy(),
        'positions': expert_obs['positions'][0].copy(),
        'position_diff': expert_obs['position_diff'][0].copy()
    }
    return [init_obs]

def train_discriminator(discriminator, expert_states, expert_actions, policy_states, policy_actions, optimizer, criterion):
    for key in expert_states:
        expert_states[key] = expert_states[key].detach()
    expert_actions = expert_actions.detach()
    for key in policy_states:
        policy_states[key] = policy_states[key].detach()
    policy_actions = policy_actions.detach()

    expert_labels = torch.ones((expert_actions.size(0), 1), dtype=torch.float32).to(discriminator.device)
    policy_labels = torch.zeros((policy_actions.size(0), 1), dtype=torch.float32).to(discriminator.device)

    expert_preds = discriminator(expert_states, expert_actions)
    policy_preds = discriminator(policy_states, policy_actions)

    loss_expert = criterion(expert_preds, expert_labels)
    loss_policy = criterion(policy_preds, policy_labels)
    loss = loss_expert + loss_policy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def collect_policy_data(policy, env, rollout_steps, device):
    obs = env.reset()
    collected_obs = {
        'ms_trades': [],
        'exchange_trades': [],
        'positions': [],
        'position_diff': []
    }
    collected_actions = []
    episodes_completed = 0
    steps = 0

    while steps < rollout_steps:
        action, _ = policy.predict(obs, deterministic=True)
        collected_actions.append(action)

        for key in collected_obs:
            collected_obs[key].append(obs[key])

        obs, _, done, infos = env.step(action)
        steps += 1

        for d in done:
            if d:
                episodes_completed += 1

        if done.any():
            obs = env.reset()

    policy_states = {
        'ms_trades': torch.tensor(collected_obs['ms_trades'][0], dtype=torch.float32).to(device),
        'exchange_trades': torch.tensor(collected_obs['exchange_trades'][0], dtype=torch.float32).to(device),
        'positions': torch.tensor(collected_obs['positions'][0], dtype=torch.float32).to(device),
        'position_diff': torch.tensor(collected_obs['position_diff'][0], dtype=torch.float32).to(device),
    }
    policy_actions_tensor = torch.tensor(collected_actions[0], dtype=torch.float32).to(device)

    return policy_states, policy_actions_tensor, episodes_completed


def main():
    # Multiple scenarios
    scenarios = [
        "data/simple_match.json",
        "data/simple_match_2.json",
        "data/simple_balance.json",
        "data/match_and_balance.json"
    ]

    # Parameters
    total_scenarios = len(scenarios)
    episodes_per_scenario = 40
    total_scenarios_to_run = len(scenarios)
    rollout_steps = 20
    policy_steps = 10000
    max_ms_trades = 3
    max_exchange_trades = 3
    action_dim = 1 + max_ms_trades + max_exchange_trades

    device = get_device("auto")

    # 1. Load all expert data
    expert_obs, expert_actions = load_multiple_expert_data(scenarios)

    # 2. Create a dummy env to initialize policy
    # use the first sample of expert_obs as initial conditions:
    initial_observations = extract_initial_observation_from_expert(expert_obs)
    dummy_env = make_vec_env(
        lambda: PositionBreakEnv(
            ms_position=initial_observations[0]['positions'][0],
            exchange_position=initial_observations[0]['positions'][1],
            ms_trades=initial_observations[0]['ms_trades'],
            exchange_trades=initial_observations[0]['exchange_trades'],
            max_ms_trades=3,
            max_exchange_trades=3
        ),
        n_envs=1
    )

    # Initialize PPO to get policy
    model = PPO("MultiInputPolicy", dummy_env, verbose=1)
    policy = model.policy

    # 3. Behavior Cloning Pretraining
    policy = behavior_cloning_pretrain(policy, expert_obs, expert_actions, device, epochs=10, batch_size=64, lr=1e-3)
    model.policy = policy

    # After BC, policy is updated with expert-like behavior.
    # 4. Proceed with GAIL training
    # Recreate env from first scenario for initial scenario training
    expert_data_path = scenarios[0]
    # We already have expert_obs from all scenarios combined, but if your training logic depends on scenario switching,
    # just use the combined sets. Or re-load if needed.

    env = make_vec_env(
        lambda: PositionBreakEnv(
            ms_position=initial_observations[0]['positions'][0],
            exchange_position=initial_observations[0]['positions'][1],
            ms_trades=initial_observations[0]['ms_trades'],
            exchange_trades=initial_observations[0]['exchange_trades'],
            max_ms_trades=3,
            max_exchange_trades=3
        ),
        n_envs=1
    )

    obs_spaces = env.observation_space.spaces
    discriminator = GAILDiscriminator(obs_spaces, action_dim, device=device).to(device)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    wrapped_env = GAILRewardWrapper(env, discriminator, device)

    # Update model's env with GAIL wrapper
    model.set_env(wrapped_env)

    current_scenario_idx = 0
    scenario_episodes_count = 0
    total_scenarios_run = 0

    # This loop is similar to your original code,
    # but now you start with a BC-pretrained policy in model.
    while total_scenarios_run < total_scenarios_to_run and current_scenario_idx < total_scenarios:
        print(f"Training scenario {current_scenario_idx+1}/{total_scenarios}")

        while scenario_episodes_count < episodes_per_scenario:
            policy_states, policy_actions_tensor, episodes_completed = collect_policy_data(
                model.policy,  # use the model's policy
                env,
                rollout_steps,
                device
            )
            scenario_episodes_count += episodes_completed

            # For simplicity, we assume expert_actions are large enough or re-sampled similarly
            # If you need repeating logic, implement similarly as previously shown.

            policy_batch_size = policy_actions_tensor.size(0)
            expert_batch_size = expert_actions.shape[0]

            batch_size = min(policy_batch_size, expert_batch_size)
            # Slice expert data
            expert_states_tensor = {
                'ms_trades': torch.tensor(expert_obs['ms_trades'][:batch_size], dtype=torch.float32, device=device),
                'exchange_trades': torch.tensor(expert_obs['exchange_trades'][:batch_size], dtype=torch.float32, device=device),
                'positions': torch.tensor(expert_obs['positions'][:batch_size], dtype=torch.float32, device=device),
                'position_diff': torch.tensor(expert_obs['position_diff'][:batch_size], dtype=torch.float32, device=device),
            }
            expert_actions_tensor_ = torch.tensor(expert_actions[:batch_size], dtype=torch.float32, device=device)

            for k,v in policy_states.items():
                policy_states[k] = v[:batch_size]
            policy_actions_tensor = policy_actions_tensor[:batch_size]

            discriminator_loss = train_discriminator(
                discriminator,
                expert_states_tensor,
                expert_actions_tensor_,
                policy_states,
                policy_actions_tensor,
                discriminator_optimizer,
                criterion
            )
            print(f"Discriminator Loss: {discriminator_loss:.4f}")

            # PPO learn step
            model.learn(total_timesteps=policy_steps, reset_num_timesteps=False)

        total_scenarios_run += 1
        scenario_episodes_count = 0

        current_scenario_idx += 1
        if current_scenario_idx < total_scenarios:
            env.close()
            # Load next scenario initial obs if needed:
            # or just continue with the combined expert data
            # If you want to switch scenario environment:
            new_initial_observations = extract_initial_observation_from_expert(expert_obs) # or use scenario-specific
            new_env = make_vec_env(
                lambda: PositionBreakEnv(
                    ms_position=new_initial_observations[0]['positions'][0],
                    exchange_position=new_initial_observations[0]['positions'][1],
                    ms_trades=new_initial_observations[0]['ms_trades'],
                    exchange_trades=new_initial_observations[0]['exchange_trades'],
                    max_ms_trades=3,
                    max_exchange_trades=3
                ),
                n_envs=1
            )

            new_wrapped_env = GAILRewardWrapper(new_env, discriminator, device)
            model.set_env(new_wrapped_env)

            print(f"Switched to scenario {current_scenario_idx+1}/{total_scenarios}")

    os.makedirs("models", exist_ok=True)
    model.save("models/gail_policy_position_break")
    torch.save(discriminator.state_dict(), "models/gail_discriminator_position_break.pth")
    print("Training complete. Models saved.")


if __name__ == "__main__":
    main()

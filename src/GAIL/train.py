import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

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
    scenarios = [
        "data/simple_match.json",
        "data/simple_match_2.json",
        "data/simple_balance.json",
        "data/match_and_balance.json"
    ]

    total_scenarios = len(scenarios)
    episodes_per_scenario = 100
    total_scenarios_to_run = len(scenarios)
    rollout_steps = 20
    policy_steps = 10000
    max_ms_trades = 3
    max_exchange_trades = 3
    action_dim = 1 + max_ms_trades + max_exchange_trades
    device = get_device("auto")

    current_scenario_idx = 0
    expert_data_path = scenarios[current_scenario_idx]
    expert_obs, expert_actions = load_expert_data_from_json(expert_data_path)

    # Move expert data to device:
    expert_states = {
        'ms_trades': torch.tensor(expert_obs['ms_trades'], dtype=torch.float32).to(device),
        'exchange_trades': torch.tensor(expert_obs['exchange_trades'], dtype=torch.float32).to(device),
        'positions': torch.tensor(expert_obs['positions'], dtype=torch.float32).to(device),
        'position_diff': torch.tensor(expert_obs['position_diff'], dtype=torch.float32).to(device),
    }
    expert_actions_tensor = torch.tensor(expert_actions, dtype=torch.float32).to(device)

    initial_observations = extract_initial_observation_from_expert(expert_obs)

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
    # Use MaskedMultiBinaryPolicy here
    policy = PPO(MaskedMultiBinaryPolicy, wrapped_env, verbose=1, tensorboard_log="./gail_tensorboard/")

    scenario_episodes_count = 0
    total_scenarios_run = 0

    while total_scenarios_run < total_scenarios_to_run and current_scenario_idx < total_scenarios:
        print(f"Training scenario {current_scenario_idx+1}/{total_scenarios}")

        while scenario_episodes_count < episodes_per_scenario:
            policy_states, policy_actions_tensor, episodes_completed = collect_policy_data(
                policy,
                env,
                rollout_steps,
                device
            )
            scenario_episodes_count += episodes_completed

            policy_batch_size = policy_actions_tensor.size(0)
            expert_batch_size = expert_actions_tensor.size(0)

            if expert_batch_size < policy_batch_size:
                # Calculate how many times we need to repeat expert data
                # to at least match the policy batch size.
                repeats = (policy_batch_size + expert_batch_size - 1) // expert_batch_size
                # For example, if policy_batch_size=50 and expert_batch_size=20,
                # repeats = (50+20-1)//20 = 69//20=3, we repeat 3 times = total 60, then slice to 50.

                # Repeat expert states and actions
                for k, v in expert_states.items():
                    # v is a tensor of shape (expert_batch_size, ...)
                    # Repeat along batch dimension
                    # Note: we keep other dimensions the same by specifying 1 for them
                    shape = [repeats] + [1] * (v.dim() - 1)
                    expanded = v.repeat(*shape)  # Now at least repeats * expert_batch_size in size
                    # Slice down to policy_batch_size
                    expert_states[k] = expanded[:policy_batch_size]

                expert_actions_repeated = expert_actions_tensor.repeat(repeats,
                                                                       1)  # (repeats*expert_batch_size, action_dim)
                expert_actions_tensor = expert_actions_repeated[:policy_batch_size]
            else:
                # If expert_batch_size >= policy_batch_size, we just slice
                batch_size = min(policy_batch_size, expert_batch_size)
                for k, v in expert_states.items():
                    expert_states[k] = v[:batch_size]
                expert_actions_tensor = expert_actions_tensor[:batch_size]

            # For policy states and actions, just slice to batch_size
            batch_size = min(policy_actions_tensor.size(0), expert_actions_tensor.size(0))  # after handling repetition
            policy_states = {k: v[:batch_size] for k, v in policy_states.items()}
            policy_actions_tensor = policy_actions_tensor[:batch_size]
            expert_states = {k: v[:batch_size] for k, v in expert_states.items()}
            expert_actions_tensor = expert_actions_tensor[:batch_size]

            discriminator_loss = train_discriminator(
                discriminator,
                expert_states,
                expert_actions_tensor,
                policy_states,
                policy_actions_tensor,
                discriminator_optimizer,
                criterion
            )
            print(f"Discriminator Loss: {discriminator_loss:.4f}")

            policy.learn(total_timesteps=policy_steps, reset_num_timesteps=False)

        total_scenarios_run += 1
        scenario_episodes_count = 0

        current_scenario_idx += 1
        if current_scenario_idx < total_scenarios:
            env.close()
            new_expert_data_path = scenarios[current_scenario_idx]
            new_expert_obs, new_expert_actions = load_expert_data_from_json(new_expert_data_path)

            # Move new expert data to device
            expert_states = {
                'ms_trades': torch.tensor(new_expert_obs['ms_trades'], dtype=torch.float32).to(device),
                'exchange_trades': torch.tensor(new_expert_obs['exchange_trades'], dtype=torch.float32).to(device),
                'positions': torch.tensor(new_expert_obs['positions'], dtype=torch.float32).to(device),
                'position_diff': torch.tensor(new_expert_obs['position_diff'], dtype=torch.float32).to(device),
            }
            expert_actions_tensor = torch.tensor(new_expert_actions, dtype=torch.float32).to(device)

            new_initial_observations = extract_initial_observation_from_expert(new_expert_obs)

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
            policy.set_env(new_wrapped_env)

            print(f"Switched to scenario {current_scenario_idx+1}/{total_scenarios}")

    os.makedirs("models", exist_ok=True)
    policy.save("models/gail_policy_position_break")
    torch.save(discriminator.state_dict(), "models/gail_discriminator_position_break.pth")
    print("Training complete. Models saved.")


if __name__ == "__main__":
    main()

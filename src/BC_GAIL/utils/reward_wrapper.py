# utils/reward_wrapper.py

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
from .preprocessing import preprocess_obs

def compute_discriminator_rewards(discriminator, obs, actions, device):
    """
    Compute GAIL rewards and add immediate penalties for invalid actions.

    Invalid actions:
      - Choose a processed trade (processed_flag=0 means processed)
      - Choose no trades at all

    Assuming:
      action structure: [action_type(1 bit), ms_trades bits, exchange_trades bits]
      and max_ms_trades=3, max_exchange_trades=3 for example.

    Adjust indexing as needed based on your environment setup.
    """
    # Preprocess observations
    states = preprocess_obs(obs)
    for k in states:
        states[k] = states[k].to(device)

    actions_array = np.array(actions)
    if actions_array.ndim == 1:
        # If we got a single action array, add batch dimension
        actions_array = actions_array[None, :]

    # Move actions to device
    actions_tensor = torch.tensor(actions_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        # GAIL reward
        rewards_tensor = -torch.log(1 - discriminator(states, actions_tensor))
        if rewards_tensor.dim() == 2 and rewards_tensor.shape[1] == 1:
            rewards_tensor = rewards_tensor.squeeze(1)

        rewards = rewards_tensor.detach().cpu().numpy()

    # Now identify invalid actions:
    # We need indexing for trades:
    # Suppose max_ms_trades and max_exchange_trades are known or can be derived
    # Let's assume them known for simplicity:
    max_ms_trades = obs['ms_trades'].shape[1]
    max_exchange_trades = obs['exchange_trades'].shape[1]

    ms_start = 1
    ms_end = ms_start + max_ms_trades
    ex_start = ms_end
    ex_end = ex_start + max_exchange_trades

    # Check processed_flag for trades:
    # obs['ms_trades'] shape: (n_envs, max_ms_trades, 5)
    # processed_flag at index 2
    ms_flags = obs['ms_trades'][:,:,2]  # shape (n_envs, max_ms_trades)
    ex_flags = obs['exchange_trades'][:,:,2] # shape (n_envs, max_exchange_trades)

    # ms_flags==1 means unprocessed (good),
    # ms_flags==0 means processed (bad if selected)

    # For indexing:
    n_envs = actions_array.shape[0]

    # Identify selected trades for each env:
    ms_selected = actions_array[:, ms_start:ms_end] # shape (n_envs, max_ms_trades)
    ex_selected = actions_array[:, ex_start:ex_end] # shape (n_envs, max_exchange_trades)

    # Check if any processed trade is selected:
    # processed means flag=0
    # We have a trade selected if corresponding bit in ms_selected/ex_selected is 1
    # invalid if selected & processed at the same position
    # Convert ms_flags, ex_flags to numpy for indexing:
    ms_flags_np = ms_flags # shape (n_envs, max_ms_trades)
    ex_flags_np = ex_flags

    # condition: processed_flag=1 means good, so processed=0 means bad
    # We want to find any selected trade where flag=0
    # invalid if: ms_selected[i,j]==1 and ms_flags_np[i,j]==0
    ms_invalid_mask = (ms_selected == 1) & (ms_flags_np == 0)
    ex_invalid_mask = (ex_selected == 1) & (ex_flags_np == 0)

    # If any processed trade selected, invalid
    processed_invalid = (ms_invalid_mask.any(axis=1)) | (ex_invalid_mask.any(axis=1))

    # Check if no trades selected:
    total_trades_selected = ms_selected.sum(axis=1) + ex_selected.sum(axis=1)
    no_trades_invalid = (total_trades_selected == 0)

    # Combine invalid conditions
    invalid_mask = processed_invalid | no_trades_invalid

    # Apply penalty for invalid actions:
    # Decide a penalty value:
    penalty = -10.0  # example large negative penalty
    rewards[invalid_mask] += penalty

    # Normalization was applied before, but now we changed rewards after that step.
    # You might choose to skip normalization after applying penalty or re-normalize.

    # If you want to re-normalize after penalty:
    if rewards.size > 1:
        std = rewards.std()
        if std > 1e-8:
            rewards = (rewards - rewards.mean()) / (std + 1e-8)

    return rewards


class GAILRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, discriminator, device):
        super(GAILRewardWrapper, self).__init__(venv)
        self.discriminator = discriminator
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Extract actions from info if needed
        actions = []
        for info in infos:
            act = info.get('action', None)
            if act is None:
                act = self.action_space.sample()
            actions.append(act)

        # Compute GAIL + penalty rewards
        rewards = compute_discriminator_rewards(self.discriminator, obs, actions, self.device)
        return obs, rewards, dones, infos

    def close(self):
        return self.venv.close()

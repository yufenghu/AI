# utils/reward_wrapper.py

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
from .preprocessing import preprocess_obs

def compute_discriminator_rewards(discriminator, obs, actions, device, processed_flag_index):
    """
    Compute GAIL rewards and add immediate penalties for invalid actions.
    Also, if action_type=match and ms_qty != ex_qty, add a penalty.

    Parameters:
    -----------
    discriminator : nn.Module
        The GAIL discriminator model.
    obs : dict
        The observation dictionary as returned by the environment.
    actions : list or np.ndarray
        The actions taken.
    device : torch.device
        The device (CPU/GPU) used for computation.
    processed_flag_index : int
        Index in the trade array where the processed_flag resides.
    """

    # Preprocess observations
    states = preprocess_obs(obs)
    for k in states:
        states[k] = states[k].to(device)

    actions_array = np.array(actions)
    if actions_array.ndim == 1:
        # If single action, add batch dim
        actions_array = actions_array[None, :]

    # Move actions to device
    actions_tensor = torch.tensor(actions_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        # GAIL reward
        rewards_tensor = -torch.log(1 - discriminator(states, actions_tensor))
        if rewards_tensor.dim() == 2 and rewards_tensor.shape[1] == 1:
            rewards_tensor = rewards_tensor.squeeze(1)

        rewards = rewards_tensor.detach().cpu().numpy()

    # Derive indexing from obs:
    max_ms_trades = obs['ms_trades'].shape[1]
    max_exchange_trades = obs['exchange_trades'].shape[1]

    # Action structure (example):
    # action[0]: action_type (0=match,1=balance,2=forceMatch)
    # action[1:1+max_ms_trades]: ms trades
    # action[1+max_ms_trades:...]: exchange trades
    action_type = actions_array[:,0].astype(int)
    ms_start = 1
    ms_end = ms_start + max_ms_trades
    ex_start = ms_end
    ex_end = ex_start + max_exchange_trades

    ms_selected = actions_array[:, ms_start:ms_end]  # (n_envs, max_ms_trades)
    ex_selected = actions_array[:, ex_start:ex_end]  # (n_envs, max_exchange_trades)

    # processed_flags from obs:
    ms_flags = obs['ms_trades'][:, :, processed_flag_index]
    ex_flags = obs['exchange_trades'][:, :, processed_flag_index]

    # Invalid if selected and processed_flag=0
    ms_invalid_mask = (ms_selected == 1) & (ms_flags == 0)
    ex_invalid_mask = (ex_selected == 1) & (ex_flags == 0)

    processed_invalid = (ms_invalid_mask.any(axis=1)) | (ex_invalid_mask.any(axis=1))

    total_trades_selected = ms_selected.sum(axis=1) + ex_selected.sum(axis=1)
    no_trades_invalid = (total_trades_selected == 0)

    invalid_mask = processed_invalid | no_trades_invalid

    # Penalty for invalid actions
    penalty = -10.0
    rewards[invalid_mask] += penalty

    # Additional penalty if action_type=match and ms_qty != ex_qty
    # We must compute ms_qty and ex_qty from obs
    # qty is at index 0 in trades
    ms_trades = obs['ms_trades']  # shape (n_envs, max_ms_trades, obs_columns)
    ex_trades = obs['exchange_trades']

    # Remember observation excludes matchgroup,balanceID but includes processed_flag and original fields
    # Ensure qty is at index 0 in the obs arrays
    # If in environment code, qty is guaranteed at index 0 of trades arrays.

    # Compute ms_qty and ex_qty selected:
    # obs arrays: (n_envs, max_ms_trades, obs_columns)
    # We must ensure qty is still at index 0 in obs columns used here.
    # If not sure, we must rely on known indexing from environment logic.
    # We'll assume qty=0 index is consistent.
    n_envs = actions_array.shape[0]

    for i in range(n_envs):
        if action_type[i] == 0:  # match
            # ms_selected_indices
            ms_selected_idx = np.where(ms_selected[i]==1)[0]
            ex_selected_idx = np.where(ex_selected[i]==1)[0]

            ms_qty_selected = obs['ms_trades'][i, ms_selected_idx, 0].sum() if ms_selected_idx.size>0 else 0
            ex_qty_selected = obs['exchange_trades'][i, ex_selected_idx, 0].sum() if ex_selected_idx.size>0 else 0

            if ms_qty_selected != ex_qty_selected:
                # Apply penalty if quantities differ
                rewards[i] += -2  # for example, a smaller penalty than invalid action
    # Optional: re-normalize or not. Here we skip re-normalization to avoid messing with carefully assigned penalties.

    return rewards


class GAILRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, discriminator, device, processed_flag_index=2):
        super(GAILRewardWrapper, self).__init__(venv)
        self.discriminator = discriminator
        self.device = device
        self.processed_flag_index = processed_flag_index

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        actions = []
        for info in infos:
            act = info.get('action', None)
            if act is None:
                act = self.action_space.sample()
            actions.append(act)

        rewards = compute_discriminator_rewards(
            self.discriminator, obs, actions, self.device, self.processed_flag_index
        )
        return obs, rewards, dones, infos

    def close(self):
        return self.venv.close()

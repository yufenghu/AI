# utils/reward_wrapper.py

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
from .preprocessing import preprocess_obs

def compute_discriminator_rewards(discriminator, obs, actions, device):
    """
    Compute rewards using the discriminator's predictions.
    This version ensures proper shaping and adds a bonus for match actions.

    Parameters:
    - discriminator (nn.Module): Trained discriminator model.
    - obs (dict): Batch of observations (contains keys like 'ms_trades', 'exchange_trades', etc.)
    - actions (list or np.ndarray): List of actions taken by the policy.
      Shape should be (n_envs, action_dim) or (action_dim,) if single environment.
    - device (torch.device): Device to perform computations on.

    Returns:
    - rewards (np.ndarray): Array of computed rewards of shape (n_envs,).
    """
    # Preprocess observations
    states = preprocess_obs(obs)
    for k in states:
        states[k] = states[k].to(device)

    # Convert actions to a numpy array and ensure (n_envs, action_dim) shape
    actions_array = np.array(actions)
    if actions_array.ndim == 1:
        # If we got a single action array, add batch dimension
        actions_array = actions_array[None, :]

    # Move actions to device
    actions_tensor = torch.tensor(actions_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        # discriminator(states, actions_tensor) should return D(s,a) probability
        # Compute GAIL reward: -log(1 - D(s,a))
        rewards_tensor = -torch.log(1 - discriminator(states, actions_tensor))

        # If rewards_tensor is (n_envs, 1), squeeze it down to (n_envs,)
        if rewards_tensor.dim() == 2 and rewards_tensor.shape[1] == 1:
            rewards_tensor = rewards_tensor.squeeze(1)

        # Now rewards_tensor should be (n_envs,)
        rewards = rewards_tensor.detach().cpu().numpy()

    # Identify match actions: action[0] == 0 means MATCH
    # actions_array is (n_envs, action_dim)
    match_mask = (actions_array[:, 0] == 1)
    # Add a bonus reward for match actions
    rewards[match_mask] += 0.1

    # Normalize rewards if there's more than one sample and std is not zero
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
        # If you need to do any special initialization, do it here.
        return obs

    def step_async(self, actions):
        # Forward the call to the wrapped environment
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Extract actions from info if needed
        actions = []
        for info in infos:
            # If actions are stored in info as per your logic
            act = info.get('action', None)
            if act is None:
                # If not available, provide a default or raise an error
                act = self.action_space.sample()
            actions.append(act)

        # Compute new rewards from the discriminator
        # Convert obs and actions to the format required by your discriminator
        rewards = compute_discriminator_rewards(self.discriminator, obs, actions, self.device)
        return obs, rewards, dones, infos


    def close(self):
        return self.venv.close()

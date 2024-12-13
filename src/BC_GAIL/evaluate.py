import os
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device

from env import PositionBreakEnv, MaskedMultiBinaryPolicy
from models import GAILDiscriminator
from utils import (
    load_expert_data_from_json,
    preprocess_obs,
    GAILRewardWrapper,
    compute_discriminator_rewards
)

def extract_initial_observation_from_expert(expert_obs):
    init_obs = {
        'ms_trades': expert_obs['ms_trades'][0].copy(),
        'exchange_trades': expert_obs['exchange_trades'][0].copy(),
        'positions': expert_obs['positions'][0].copy(),
        'position_diff': expert_obs['position_diff'][0].copy()
    }
    return [init_obs]

def evaluate_policy(policy, env, n_eval_episodes=5):
    episode_rewards = []
    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward}")
    print(f"Mean Reward over {n_eval_episodes} episodes: {np.mean(episode_rewards)}")

def main():
    # scenario = "data/simple_match.json"
    # scenario = "data/simple_balance.json"
    scenario = "data/match_and_balance_test.json"

    policy_path = "models/gail_policy_position_break"
    discriminator_path = "models/gail_discriminator_position_break.pth"
    device = get_device("auto")

    expert_obs, expert_actions = load_expert_data_from_json(scenario)
    initial_observations = extract_initial_observation_from_expert(expert_obs)

    max_ms_trades = 3
    max_exchange_trades = 3

    env = make_vec_env(
        lambda: PositionBreakEnv(
            ms_position=initial_observations[0]['positions'][0],
            exchange_position=initial_observations[0]['positions'][1],
            ms_trades=initial_observations[0]['ms_trades'],
            exchange_trades=initial_observations[0]['exchange_trades'],
            max_ms_trades=max_ms_trades,
            max_exchange_trades=max_exchange_trades
        ),
        n_envs=1
    )

    obs_spaces = env.observation_space.spaces
    action_dim = 1 + max_ms_trades + max_exchange_trades

    discriminator = GAILDiscriminator(obs_spaces, action_dim, device=device).to(device)
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    discriminator.eval()

    wrapped_env = GAILRewardWrapper(env, discriminator, device)

    # Load policy with custom policy class
    policy = PPO.load(policy_path, device=device, env=wrapped_env)
    policy.set_env(wrapped_env)

    # Evaluate the policy
    evaluate_policy(policy, wrapped_env, n_eval_episodes=5)

if __name__ == "__main__":
    main()

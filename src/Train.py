from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Correct import path
import numpy as np
from PositionBreakEnv import PositionBreakEnv


def train_agent(ms_position=15, exchange_position=20, ms_trades=None, exchange_trades=None):
    # Create the environment with dynamic positions and trades
    env = PositionBreakEnv(ms_position=ms_position, exchange_position=exchange_position, ms_trades=ms_trades,
                           exchange_trades=exchange_trades)

    # Wrap the environment in DummyVecEnv for Stable-Baselines3
    env = DummyVecEnv([lambda: env])  # Ensure the environment is wrapped

    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_position_break_tensorboard/")

    # Train the model
    model.learn(total_timesteps=100000)  # Adjust the number of timesteps as needed

    # Save the trained model
    model.save(f"ppo_position_break_{ms_position}_{exchange_position}")

    return model


# Example: Train with custom MS and Exchange positions, and trades
ms_trades = np.array([
            [5, 100], [5, 102], [5, 101]
        ], dtype=np.float32)
exchange_trades = np.array([
            [5, 98], [10, 99], [5, 101]
        ], dtype=np.float32)

# Train the agent with custom initial positions and trades
model = train_agent(ms_position=15, exchange_position=20, ms_trades=ms_trades, exchange_trades=exchange_trades)

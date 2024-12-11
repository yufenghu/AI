import numpy as np
from stable_baselines3 import PPO
from PositionBreakEnv import PositionBreakEnv


if __name__ == "__main__":
    # Example trades:
    # MS trades: one trade with matchgroup=1
    ms_trades = np.array([
        [5, 103, 1, 1, 0],
        [5, 103, 1, 1, 0],
        [5, 103, 1, 1, 0]  # qty=10, price=100, not processed, matchgroup=1, balanceID=-1
    ], dtype=np.float32)

    # Exchange trades: three trades, two with matchgroup=1, one with balanceID=1
    exchange_trades = np.array([
        [5, 103, 1, 1, 0],
        [10, 103, 1, 1, 0],  #
        [5, 100, 1, 0, 1]
    ], dtype=np.float32)

    env = PositionBreakEnv(ms_position=10.0, exchange_position=15.0,
                           ms_trades=ms_trades, exchange_trades=exchange_trades, max_ms_trades=3, max_exchange_trades=3)

    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=5, batch_size=5, ent_coef=0.9, gamma=0.9,learning_rate=0.005, tensorboard_log="./ppo_tensorboard/", device="cuda")
    model.learn(total_timesteps=300000)

    model.save("position_break_model")

import numpy as np
from stable_baselines3 import PPO
from PositionBreakEnv import PositionBreakEnv

if __name__ == "__main__":
    # Same trades as training
    # ms_trades = np.array([
    #     [5, 100, 1, 1, 0]
    # ], dtype=np.float32)
    #
    # exchange_trades = np.array([
    #     [5, 102, 1, 1, 0]
    # ], dtype=np.float32)

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

    # Load the previously trained model
    model = PPO.load("position_break_model")

    # Run a test episode until all trades are processed
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done:
        # Use the trained model to predict the next action
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()  # Optional: visualize or print the environment state
        print(f"Step: {step_count}, Action: {action}, Reward: {reward}, Done: {done}")
        step_count += 1

    print("Episode finished!")
    summary = env.get_action_summary()
    print("Action summary:")
    for record in summary:
        print(record)

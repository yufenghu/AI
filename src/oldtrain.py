from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from PositionBreakEnv import PositionBreakEnv

# Create and check the environment
env = PositionBreakEnv()
check_env(env)

# Train the model using PPO
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Test the trained model
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print("Action:", action)
    print("Reward:", reward)
    print("Done:", done)

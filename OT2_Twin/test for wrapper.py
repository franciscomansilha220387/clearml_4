# %%
from stable_baselines3.common.env_checker import check_env
from ot2_env_wrapper_2 import OT2Env

# %%
# instantiate your custom environment
wrapped_env = OT2Env(render=True) # modify this to match your wrapper class

# %%
# Assuming 'wrapped_env' is your wrapped environment instance
check_env(wrapped_env)

# %%
import gymnasium as gym
import numpy as np

# Load your custom environment
env = wrapped_env

# Number of episodes
num_episodes = 1

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break



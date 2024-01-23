import gymnasium as gym
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO
from clearml import Task
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
 
physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')

#conda_env = "gpu1"
# Activate the Anaconda environment using the conda command
#os.system(f"conda activate {conda_env}")

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# initialize wandb project
run = wandb.init(project="RL model",sync_tensorboard=True)

#task = Task.init(project_name='Mentor Group D/Group 3', task_name='Experiment1')


#copy these lines exactly as they are
#setting the base docker image
#task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
#task.execute_remotely(queue_name="default")

# Create the environment
env = OT2Env(render=False , max_steps=1000)

#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--learning_rate", type=float, default=0.0003)
#parser.add_argument("--batch_size", type=int, default=64)
#parser.add_argument("--n_steps", type=int, default=2048)
#parser.add_argument("--n_epochs", type=int, default=10)

#args = parser.parse_args()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")
            #learning_rate=args.learning_rate, 
            #batch_size=args.batch_size, 
            #n_steps=args.n_steps, 
            #n_epochs=args.n_epochs, 
            

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=100000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# Test the environment
obs, info = env.reset()

# Train the model in aloop and save the weights incrementally
for i in range(10):
    model.learn(total_timesteps=100000, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")
    model.save(f'./RL_models3/rl_model_{i}')
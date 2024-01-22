import gymnasium as gym
import time
import os
from ot2_env_wrapper_2 import OT2Env
from simple_pid import PID
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# instantiate your custom environment
wrapped_env = OT2Env(render=True) 

# Initialize PID controllers for each axis

pid_x = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=0)
pid_y = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=0)
pid_z = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=0)

pid_x.setpoint = 0.1439
pid_y.setpoint = 0.1603
pid_z.setpoint = 0.1195

# Limits for each axis based on the robot's working envelope from task 9
a = {'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}

pid_x.output_limits = (-0.1871, 0.253)
pid_y.output_limits = (-0.1706, 0.2195)
pid_z.output_limits = (0.1197, 0.2898)

# Number of iterations for the PID control loop
num_iterations = 1000
threshold = 0.01  # Threshold for considering the position to be reached

for i in range(num_iterations):
    # Retrieve the current pipette position from the environment
    observation, _ = wrapped_env.reset()
    current_position = observation[:3]  # Assuming the first 3 elements are x, y, z

    # Calculate the PID control for each axis
    control_x = pid_x(current_position[0])
    control_y = pid_y(current_position[1])
    control_z = pid_z(current_position[2])

    # Construct the action and apply it to the environment
    action = np.array([control_x, control_y, control_z])
    new_observation, reward, terminated, truncated, _ = wrapped_env.step(action)

    # Check if the setpoint is reached within the threshold
    new_position = new_observation[:3]
    if np.linalg.norm(new_position - np.array([pid_x.setpoint, pid_y.setpoint, pid_z.setpoint])) < threshold:
        print(f"Target position reached in {i+1} iterations.")
        break

    # Optionally, you can also check for termination based on 'terminated' or 'truncated'

# Close the environment when done
wrapped_env.close()

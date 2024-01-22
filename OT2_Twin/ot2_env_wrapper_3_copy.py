import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation





class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.render_mode = render

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Box(-1, 1, (3,), np.float32)

        # Define observation space with only pipette_position
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        self.previous_distance = 1

        # keep track of the number of steps
        self.steps = 0

        working_envelope = {'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}

        # Generate random values within the specified range for each axis
        random_x = np.random.uniform(working_envelope['min_x'], working_envelope['max_x'])
        random_y = np.random.uniform(working_envelope['min_y'], working_envelope['max_y'])
        random_z = np.random.uniform(working_envelope['min_z'], working_envelope['max_z'])

        # Create the random goal position as a NumPy array
        self.goal_position = np.array([random_x, random_y, random_z], dtype=np.float32)





    def reset(self, seed=None):
            # being able to set a seed is required for reproducibility
            if seed is not None:
                np.random.seed(seed)

            observation = self.sim.reset()

            # Calculate current distance to goal  
            self.previous_distance = 1

            # Reset the state of the environment to an initial state
            # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
                
            working_envelope = {'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}

            # Generate random values within the specified range for each axis
            random_x = np.random.uniform(working_envelope['min_x'], working_envelope['max_x'])
            random_y = np.random.uniform(working_envelope['min_y'], working_envelope['max_y'])
            random_z = np.random.uniform(working_envelope['min_z'], working_envelope['max_z'])

            # Create the random goal position as a NumPy array
            self.goal_position = np.array([random_x, random_y, random_z], dtype=np.float32)

            robot_id = list(observation.keys())[0]

            observation = np.array(observation[robot_id]['pipette_position'], dtype=float)

            # Append the goal position to the pipette position
            observation = np.hstack((observation, self.goal_position), dtype=np.float32)
            
            # Reset the number of steps
            self.steps = 0

            inf = {}

            return observation, inf
    

    def step(self, action):
            # Execute one time step within the environment
            # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
            action = np.append(action, 0.0)

            # Call the environment step function
            observation = self.sim.run([action]) # Why do we need to pass the action as a list? Because the simulation class expects a list of actions

            robot_id = list(observation.keys())[0]

            observation = np.array(observation[robot_id]['pipette_position'], dtype=float)

            # Append the goal position to the pipette position
            observation = np.hstack((observation, self.goal_position), dtype=np.float32)

            # Calculate current distance to goal
            distance_to_goal = np.linalg.norm(observation[:3] - self.goal_position)

            # Calculate improvement
            improvement = self.previous_distance - distance_to_goal

            # Update previous distance
            self.previous_distance = distance_to_goal

            # Define reward function
            if distance_to_goal < 0.01:  # Goal threshold (adjust as needed)
                reward = 1000  # Large reward for reaching the goal
                terminated = True
            else:
                reward = improvement*5  # Reward is proportional to improvement
                terminated = False

            # Check for truncation
            truncated = self.steps >= self.max_steps

            # Increment step count
            self.steps += 1

            info = {}  # Additional info (if any)

            return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        # Check if rendering is enabled
        if self.render_mode:
            # Implement the rendering logic here
            # For example, visualize the environment's state
            pass
        else:
            # If rendering is disabled, do nothing or print a message
            print("Rendering is disabled.")


    def close(self):
        self.sim.close()
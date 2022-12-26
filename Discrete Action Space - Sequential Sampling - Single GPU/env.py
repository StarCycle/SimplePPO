import random
import numpy as np
import gym
from gym import spaces

class Grid1DEnv(gym.Env):
  """
  An environment eith Gym interface
  There are grids to fill. 
  If the RL agent fill a blank grid, it gets a reward of +1, otherwise the reward is -1
  """

  def __init__(self, grid_size, num_epoch_steps):
    super(Grid1DEnv, self).__init__()
    self.max_num_steps = num_epoch_steps
    self.current_step = 0

    # Define action and observation space. They must be gym.spaces objects
    n_actions = grid_size
    self.state = [0]*grid_size
    self.action_space = spaces.Discrete(n_actions)
    self.observation_space = spaces.Box(low=0, high=grid_size,shape=(grid_size,), dtype=np.float32)

  def seed(self, seed):
    random.seed(seed)

  def reset(self):
    """
    Important: the observation must be a numpy array
    """
    self.state = [0]*len(self.state)
    self.state[random.randrange(0, len(self.state))] = 1
    self.current_step = 0
    return self.state, {}

  def step(self, action):
    if action < 0 or action >= len(self.state):
        raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    reward = 1 if self.state[action] == 0 else 0
    self.state[action] = 1
    self.current_step += 1
    if self.current_step >= self.max_num_steps:
        self.reset()
        return self.state, reward, True, None, {}
    return self.state, reward, False, None, {}
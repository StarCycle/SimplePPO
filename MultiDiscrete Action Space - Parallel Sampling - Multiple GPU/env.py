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

  def __init__(self, seed, grid_size, action_space, num_epoch_steps):
    super(Grid1DEnv, self).__init__()
    random.seed(seed)
    self.max_num_steps = num_epoch_steps
    self.current_step = 0

    # Define action and observation space. They must be gym.spaces objects
    n_actions = grid_size
    self.state = [0]*grid_size
    self.action_space = spaces.MultiDiscrete(action_space)
    self.observation_space = spaces.Box(low=0, high=grid_size,shape=(grid_size,), dtype=np.float32)

  def reset(self):
    """
    Important: the observation must be a numpy array
    """
    self.state = [0]*len(self.state)
    self.state[random.randrange(0, len(self.state))] = 1
    self.current_step = 0
    return self.state, {}

  def step(self, action):
    reward = 0
    for actionID in action:
      if self.state[actionID] == 0:
        reward += 1
        self.state[actionID] = 1
      elif self.state[actionID] == 1:
        reward -= 1
    self.current_step += 1
    if self.current_step >= self.max_num_steps:
        self.reset()
        return self.state, reward, True, None, {}
    return self.state, reward, False, None, {}
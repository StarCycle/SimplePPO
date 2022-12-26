# Simple PPO
An extremely simple PPO implementation based on Pytorch with ~270 lines of code. It supports:

 - Grid search of hyperparameters
 - Showing curves with tensorboard
 - Parallel sampling from multiple environments via Gym interface. You can define your own parallel environment by modifying `env.py`
 - Discrete action space (select an action in a step) and MultiDiscrete action space (select multiple actions in a step).
 - Single GPU or multi-GPU training via nn.DataParallel of Pytorch.

This implementation is a simplification of the PPO algorithm in cleanRL ([link](https://github.com/vwxyzjn/cleanrl)). 

### How to use it?

Just run the `ppo.py` in each folder. 
During training, the output will be recorded in a `runs` folder. You can visualize the output by:

    tensorboard --logdir=runs

### Note

 - `env.py` contains a "filling grid" environment. There are several grids. If the agent fills a blank grid, it will receive a reward of +1, otherwise the reward will be -1.
 - When you use multiple GPUs, the number of GPUs should be smaller or equal to the number of parallel environments.
 - I am using `gym.vector.AsyncVectorEnv` to create parallel environments with multiprocessing. However, debugging a  multiprocessing program is complicated. Thus, I advise you to switch to `gym.vector.SyncVectorEnv` during debugging, which only uses multithreading.
 - Current multi-GPU capability can work but is quite slow. I will improve it.

### Citation

Please cite the following work:

_Li, Z. (2022). Use Reinforcement Learning to Generate Testing Commands for Onboard Software of Small Satellites._

The RL algorithms in this work are in [StarCycle/TestCommandGeneration](https://github.com/StarCycle/TestCommandGeneration)


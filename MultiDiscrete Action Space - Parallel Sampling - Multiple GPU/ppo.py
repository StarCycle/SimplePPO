import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gym
from env import Grid1DEnv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, action_space, num_nn, critic_std, actor_std):
        super().__init__()
        self.action_space = action_space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape[1]).prod(), num_nn)), 
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, 1), std=critic_std),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape[1]).prod(), num_nn)), 
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, num_nn)),
            nn.ReLU(),
            layer_init(nn.Linear(num_nn, sum(action_space)), std=actor_std),
        )

    def forward(self, x, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.action_space, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals]).T
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action.T, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals]).T
        return action, logprob.sum(0), entropy.sum(0), self.critic(x)

def train(envs, name, num_envs, action_space, target_kl, minibatch_size, gamma, ent_coef, vf_coef, num_nn, critic_std, actor_std, learning_rate, num_epoch_steps):

    # These hyperparameters are not considered in grid search
    total_timesteps = 200000     # How many steps you interact with the env
    num_env_steps = 128           # How many steps you interact with the env before an update
    num_update_steps = 4         # How many times you update the neural networks after interation
    gae_lambda = 0.95               # Parameter in advantage estimation
    max_grad_norm = 0.5           # max norm of the gradient vector
    clip_coef = 0.2                      # Parameter to clip the (p_new/p_old) ratio

    writer = SummaryWriter('runs/' + name)
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    agent = Agent(envs, action_space, num_nn, critic_std, actor_std).to(devices[0])
    agent = nn.DataParallel(agent, device_ids=devices)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize storage for a round
    obs = torch.zeros((num_env_steps, num_envs) + envs.single_observation_space.shape).to(devices[0])
    actions = torch.zeros((num_env_steps, num_envs, len(action_space))).to(devices[0])
    logprobs = torch.zeros((num_env_steps, num_envs)).to(devices[0])
    rewards = torch.zeros((num_env_steps, num_envs)).to(devices[0])
    dones = torch.zeros((num_env_steps, num_envs)).to(devices[0])
    values = torch.zeros((num_env_steps, num_envs)).to(devices[0])
    next_obs = torch.Tensor(envs.reset()[0]).to(devices[0]) 
    next_done = torch.zeros(num_envs).to(devices[0])	

    global_step = 0
    cumu_rewards = 0
    num_rounds = total_timesteps // num_env_steps
    for round in range(num_rounds):

        # action logic
        for step in range(num_env_steps):
            global_step += 1*num_envs
            obs[step] = next_obs
            dones[step] = next_done
                            
            with torch.no_grad():
                action, logprob, _, value = agent.forward(next_obs.reshape((num_envs, -1)))
            values[step] = value.flatten()           
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, done, _, info = envs.step(action.cpu()) 
            cumu_rewards += reward
            for i in range(num_envs):
                if done[i] == True:
                    writer.add_scalar("cumulative rewards", cumu_rewards[i], global_step)
                    print("global step:", global_step, "cumulative rewards:", cumu_rewards[i])
                    cumu_rewards[i] = 0
            rewards[step] = torch.tensor(reward).to(devices[0]).view(-1) 
            next_obs, next_done = torch.Tensor(next_obs).to(devices[0]), torch.Tensor(done).to(devices[0])
	
        # bootstrap value if not done
        with torch.no_grad():
            _, _, _, next_value = agent.forward(next_obs)
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(devices[0])
            lastgaelam = 0 
            for t in reversed(range(num_env_steps)):
                if t == num_env_steps - 1:
                    nextnonterminal = 1.0 - next_done 
                else:
                    nextnonterminal = 1.0 - dones[t + 1] 
                    next_value = values[t + 1]
                delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, envs.observation_space.shape[1]))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, len(action_space)))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        inds = np.arange(num_env_steps)		
        for update in range(num_update_steps):
            clipfracs = []
            approx_kl = []
            np.random.shuffle(inds)
            for start in range(0, num_env_steps, minibatch_size):
                end = start + minibatch_size
                minds = inds[start:end]
                _, newlogprob, entropy, newvalue = agent.forward(b_obs[minds], b_actions[minds])
                logratio = newlogprob - b_logprobs[minds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl += [((ratio - 1) - logratio).mean().item()]
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                # Policy loss
                pg_loss1 = - b_advantages[minds] * ratio
                pg_loss2 = - b_advantages[minds] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) 
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[minds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # Update the neural networks
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            # Annealing the learning rate, if KL is too high
            if np.mean(approx_kl) > target_kl:
                optimizer.param_groups[0]["lr"] *= 0.99
		
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("value_loss", v_loss.item(), global_step)
        writer.add_scalar("policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("entropy", entropy_loss.item(), global_step)
        writer.add_scalar("approx_kl", np.mean(approx_kl), global_step)
        writer.add_scalar("clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("explained_variance", explained_var, global_step)
        writer.add_scalar("mean_value", values.mean().item(), global_step)
    writer.close()

def make_env(seed, grid_size, action_space, num_epoch_steps):
    def thunk():
        env = Grid1DEnv(seed, grid_size, action_space, num_epoch_steps)
        return env
    return thunk

if __name__ == "__main__":

    target_kl = [0.02]	
    minibatch_size = [32]	# The batch size to update the neural network
    gamma = [0.9]
    ent_coef = [0.001]	# Weight of the entropy loss in the total loss
    vf_coef = [0.5]	    	# Weight of the value loss in the total loss
    num_nn = [512]
    critic_std = [1]
    actor_std = [0.01]
    learning_rate = [5e-4]
    num_epoch_steps = 7
    grid_size = 16
    action_space = [16, 16]	# Action 1 can be selected from [0,15], action 2 from [0,15]
    num_envs = 4
    env_seed = 1		

    envs = gym.vector.AsyncVectorEnv(
	[make_env(env_seed+i, grid_size, action_space, num_epoch_steps) for i in range(num_envs)])

    # Grid search of part of hyperparameters
    for tk in target_kl:
        for bs in minibatch_size:
            for ga in gamma:
                for ef in ent_coef:
                    for vf in vf_coef:
                        for num in num_nn:
                            for cstd in critic_std:
                                for astd in actor_std:
                                    for lr in learning_rate:
                                        name = 'tk'+str(tk)+'_bs'+str(bs)+'_ga'+str(ga)+'_ef'+str(ef)+'_vf'+str(vf)+'_num'+str(num)+'_cs'+str(cstd)+'_as'+str(astd)+'_lr'+str(lr)
                                        train(envs, name, num_envs, action_space, tk, bs, ga, ef, vf, num, cstd, astd, lr, num_epoch_steps)

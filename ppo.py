import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import gym
import cv2 as cv
from torch.distributions import Categorical

device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# Critic module
class Critic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

"""class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.model(x)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.model(x)"""



class PPO:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n
        self.timesteps_per_batch = 500  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.00025  # Learning rate of actor optimizer
        self.gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.1  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None
        # initialize actor critic networks
        self.actor = Actor(self.obs_dim,self.act_dim)
        self.critic = Critic(self.obs_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.00025)


    def get_action(self, obs):
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.detach(), logprob.detach()

    def learn(self, num_epochs):

        for epoch in tqdm(range(num_epochs)):
            batch_obs, next_obs, batch_acts, batch_probs, batch_rtgs, batch_lens, dones,curr_reward, rewards = self.rollout()
            V = self.evaluate_v(batch_obs)
            A_k = batch_rtgs-V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            target = curr_reward   + self.gamma *self.evaluate_v(next_obs).detach()*(1-dones)
            for _ in range(self.n_updates_per_iteration):
                V = self.evaluate_v(batch_obs)
                curr_probs = self.evaluate_prob(batch_obs, batch_acts)
                ratios = torch.exp(curr_probs-batch_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.SmoothL1Loss()(V, target)
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            if epoch!=0 and epoch%10==0:
                print("epoch {} , average score = {}".format(epoch,  np.mean(rewards)))






    def rollout(self):

        # Number of timesteps run so far this batch
        t = 0
        #i = 0
        batch_lens = []
        batch_rews = []
        rewards = []
        ep_rews = []
        batch_obs = []
        next_obs = []
        batch_acts = []
        batch_probs = []
        curr_reward = []
        dones = []

        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            #i+=1

            obs = self.env.reset()
            done = False
            ep_t = 0
            while True:
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action.item())

                # Collect reward, action, and log prob
                curr_reward.append(rew)
                dones.append(done)
                next_obs.append(obs)
                ep_rews.append(rew)
                batch_acts.append(action.item())
                batch_probs.append(prob)
                if done:
                    break
                ep_t+=1
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            rewards.append(np.sum(ep_rews))

        # Reshape data as tensors in the shape specified before returning
        #print(i)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        next_obs = torch.tensor(next_obs, dtype=torch.float)
        curr_reward = torch.tensor(curr_reward, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_probs = torch.tensor(batch_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, next_obs, batch_acts, batch_probs, batch_rtgs, batch_lens, dones, curr_reward, rewards

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs


    def evaluate_v(self, batch_obs):
        return self.critic(batch_obs).squeeze()

    def evaluate_prob(self, batch_obs, batch_acts):
        probs = self.actor(batch_obs)
        dist = Categorical(probs)
        logprobs = dist.log_prob(batch_acts)
        probs = torch.log(probs.gather(1, batch_acts.unsqueeze(1).long()).squeeze())
        return probs

class SkipFrames(gym.Wrapper):
    """The successive observations will be similar, so we will take every skip observation"""
    def __init__(self,env,skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self,action):

        total_reward = 0
        for _ in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info

    def reset(self):

        observation = self.env.reset()
        return observation

class PreProcessing(gym.ObservationWrapper):

    def __init__(self,env,NumFrames):
        super().__init__(env)
        self.NumFrames = NumFrames
        self.observation_space = gym.spaces.Box(low = 0.0,high = 1.0,shape=(NumFrames,84,84), dtype=np.float32)
        self.observations = np.zeros((NumFrames,84,84),dtype=np.float32)

    def observation(self, observation):

        observation = cv.cvtColor(observation,cv.COLOR_RGB2GRAY)
        observation = cv.resize(observation, (84, 110), interpolation=cv.INTER_AREA)
        observation = observation[18:102,:]
        observation = np.reshape(observation,[1,84,84]).astype(np.float32)/255.0
        self.observations[:-1] = self.observations[1:]
        self.observations[-1] = observation
        return self.observations
    def reset(self):
        self.observations = np.zeros((self.NumFrames, 84, 84),dtype=np.float32)
        return self.observation(self.env.reset())

env = gym.make('Breakout-v0')
env = SkipFrames(env)
env = PreProcessing(env,4)
#env = gym.make("CartPole-v1")
my_ppo = PPO(env)
my_ppo.learn(500)












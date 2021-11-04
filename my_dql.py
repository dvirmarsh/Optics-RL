import numpy as np
import gym
import cv2 as cv
import torch.nn as nn
import torch
from tqdm import tqdm
import random
import collections

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


class DqnNet(nn.Module):

    def __init__(self,in_shape,num_actions):
        super().__init__()
        """self.ConvNet = nn.Sequential(
            nn.Conv2d(in_shape[0],16,6,3),
            nn.ReLU(),
            nn.Conv2d(16,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
        )"""
        self.ConvNet2 = nn.Sequential(
            nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        out_size = self.ConvNet2(torch.zeros(1,*in_shape)).shape
        self.DqnLinear = nn.Sequential(
            nn.Linear(np.prod(out_size), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        x = self.ConvNet2(x)
        x = x.view(x.size(0), -1)
        return self.DqnLinear(x)

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DQN network
        self.dqn = DqnNet(state_space, action_space).to(self.device)

        if self.pretrained:
            self.dqn.load_state_dict(torch.load("DQN.pt", map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            with open("ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

class MyAgent:

    def __init__(self, in_shape, num_actions, lr, gamma, epsilon0, epsilon_decay, epsilon_min, max_memory=30000, batch_size=32):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = DqnNet(in_shape, num_actions).to(self.device)
        self.num_actions = num_actions
        self.rewards = torch.zeros((max_memory, 1), dtype=torch.float32)
        self.states = torch.zeros((max_memory, *in_shape), dtype=torch.float32)
        self.actions = torch.zeros((max_memory, 1), dtype=torch.float32)
        self.dones = torch.zeros((max_memory, 1), dtype=torch.float32)
        self.next_states = torch.zeros((max_memory, *in_shape), dtype=torch.float32)
        self.loss = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.idx = 0
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.num_in_queue = 0

    def pick_action(self,state):
        if random.random()<self.epsilon:
            return torch.tensor([[random.randrange(self.num_actions)]])
        else:
            return torch.argmax(self.net(torch.tensor(state.to(self.device)))).unsqueeze(0).unsqueeze(0).cpu()
    def  save(self,state,action,reward,next_state,done):
        self.rewards[self.idx] = reward.float()
        self.states[self.idx] = state.float()
        self.actions[self.idx] = action.float()
        self.dones[self.idx] = done.float()
        self.next_states[self.idx] = next_state.float()
        self.idx = (self.idx + 1) % self.max_memory
        self.num_in_queue = min(self.num_in_queue+1,self.max_memory)
    def update(self):
        idx = random.choices(range(self.num_in_queue), k=self.batch_size)
        STATE = self.states[idx].to(self.device)
        ACTION = self.actions[idx].to(self.device)
        REWARD = self.rewards[idx].to(self.device)
        STATE2 = self.next_states[idx].to(self.device)
        DONE = self.dones[idx].to(self.device)
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.net(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.net(STATE).gather(1, ACTION.long())

        loss = self.loss(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)


def my_run(num_episodes):
    env = gym.make('Breakout-v0')
    env = SkipFrames(env)
    env = PreProcessing(env, 4)
    in_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = MyAgent(in_shape, num_actions,  lr=0.00025, gamma=0.9, epsilon0=1, epsilon_decay=0.99, epsilon_min=0.02)
    total_rewards = []
    steps = 0
    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.tensor([state])
        total_reward = 0

        while True:
            action = agent.pick_action(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)


            agent.save(state, action, reward, state_next, terminal)
            if steps>=32:
                agent.update()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 50 == 0:
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                     np.mean(total_rewards)))





def run(training_mode, pretrained, num_episodes=1000, exploration_max=1):
    env = gym.make('Breakout-v0')  # can change the environmeent accordingly
    #env = gym.make('MsPacman-v0')
    #env = create_env(env)  # Wraps the environment so that frames are grayscale
    env = SkipFrames(env)
    env = PreProcessing(env,4)

    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.2,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     pretrained=pretrained)

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset()

    total_rewards = []
    if training_mode and pretrained:
        with open("total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 50 == 0:
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                     np.mean(total_rewards)))
        num_episodes += 1

    print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

        torch.save(agent.dqn.state_dict(), "DQN.pt")
        torch.save(agent.STATE_MEM, "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    env.close()


run(training_mode = True, pretrained = False, num_episodes=1000, exploration_max=1)
#my_run(num_episodes=1000)


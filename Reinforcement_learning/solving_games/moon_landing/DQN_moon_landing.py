# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
from collections import deque, namedtuple
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v2', render_mode = "human")


env.reset()

# Parameters
N_episodes = 700                            # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
epsilon_max = 0.99
epsilon_min = 0.05
epsilon = 0.99
end_decay = int(N_episodes * 0.90)
batch_size = 64
buffer_size = 30_000
learning_rate = 5E-4
actualise_target = buffer_size//batch_size

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

# Actualise network
def update_target_network(target, eval_net):
    target.load_state_dict(eval_net.state_dict())

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
class ExperienceReplayBuffer:
    """
    Taken from exercise session 3!

    Replay buffer for storing experiences.
    
       The experience replay buffer stores past experiences so that the agent can learn from them later.
       By sampling randomly from these experiences, the agent avoids overfitting to the most recent 
       transitions and helps stabilize training.
       - The buffer size is limited, and older experiences are discarded to make room for new ones.
       - Experiences are stored as tuples of (state, action, reward, next_state, done).
       - A batch of experiences is sampled randomly during each training step for updating the Q-values."""
    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)  # Using deque ensures efficient removal of oldest elements

    def append(self, experience):
        """Add a new experience to the buffer"""
        self.buffer.append(experience)

    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def sample_batch(self, n):
        """Randomly sample a batch of experiences"""
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        indices = np.random.choice(len(self.buffer), size=n, replace=False)  # Random sampling
        batch = [self.buffer[i] for i in indices]  # Create a batch from sampled indices
        return zip(*batch)  # Unzip batch into state, action, reward, next_state, and done

class NetworkDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Define forward pass"""
        x = self.activation(self.input_layer(x))  # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x))  # Apply hidden layer and ReLU
        return self.output_layer(x)  # Return Q-values for all actions

class NetworkDQNDuelling(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.advantage_layer = nn.Linear(64, output_size)
        self.value_layer = nn.Linear(64, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Define forward pass"""
        x = self.activation(self.input_layer(x))  # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x))  # Apply hidden layer and ReLU
        advantage = self.advantage_layer(x)
        value = self.value_layer(x)
        advantage_average = torch.mean(advantage, dim=1, keepdim=True)
        return value + advantage - advantage_average # Return Q-values for all actions with duelling DQN

# initialize buffer
experience_buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
# initialise network
network = NetworkDQNDuelling(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
target_network = NetworkDQNDuelling(input_size=env.observation_space.shape[0], output_size=env.action_space.n)

# optimizer
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
target_network.load_state_dict(network.state_dict())


def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        return network(state_tensor).max(1)[1].item()
    
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    epsilon = max(epsilon_min, epsilon_max-(epsilon_max-epsilon_min)*(i-1)/(end_decay-1)) # linear
    # epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**((i)/end_decay)) # exponential
    while not (done or truncated):
        # Take a random action
        action = select_action(state, epsilon)
        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)
        # Update episode reward
        total_episode_reward += reward
        experience_buffer.append(Experience(state, action, reward, next_state, done))
        # Update state for next iteration
        state = next_state
        t+= 1
        if len(experience_buffer) > batch_size:
            states, actions, rewards, next_states, dones = experience_buffer.sample_batch(batch_size)
            states = torch.tensor(states, dtype=torch.float32, requires_grad=True)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # # CER
            # states = torch.cat((states, torch.tensor([state], dtype=torch.float32)), dim=0)
            # actions = torch.cat((actions, torch.tensor([action], dtype=torch.int64))).unsqueeze(1)
            # next_states = torch.cat((next_states, torch.tensor([next_state], dtype=torch.float32)),dim=0)
            # rewards = torch.cat((rewards, torch.tensor([reward], dtype=torch.float32)))
            # dones = torch.cat((dones, torch.tensor([done],dtype=torch.float32)))
            q_values = network(states).gather(dim=1, index=actions).squeeze()
            # compute target values:
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0].detach()
                targets = rewards + discount_factor*next_q_values*(1-dones) # if terminal states then we don't change the Q value
            
            loss = nn.functional.mse_loss(q_values,targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)
            optimizer.step()
            #actualise the target
            if t%actualise_target == 0:
                update_target_network(target_network, network)
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - Epsilon: {:.2f}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1], epsilon))
    
    #break if early convergence
    if running_average(episode_reward_list, n_ep_running_average)[-1] > 100:
        break

# Close environment
env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, len(running_average(
    episode_reward_list, n_ep_running_average))+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, len(running_average(
    episode_number_of_steps, n_ep_running_average))+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

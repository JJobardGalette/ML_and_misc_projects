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
from DDPG_agent import RandomAgent
import warnings

from DDPG_soft_updates import soft_updates
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

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLanderContinuous-v2', render_mode = "human")

env.reset()
# Parameters
N_episodes = 3000               # Number of episodes to run for training
discount_factor = 0.99         # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
m = len(env.action_space.high) # dimensionality of the action
buffer_size = 30_000
noise = 0
batch_size = 64
policy_update_freq = 2
# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# learning rates
learning_rate_critic = 5E-4
learning_rate_actor = 5E-5

# Agent initialization
agent = RandomAgent(m)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# Defining noise
def compute_next_noise(previous_noise, mu=0.15, sigma=0.2):
    return -mu*previous_noise+np.random.normal(0, sigma**2,m)



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

class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size=2):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 400)
        self.hidden_layer = nn.Linear(400, 200)
        self.output_layer = nn.Linear(200, output_size)
        self.activation = nn.ReLU()
        self.activation_out = nn.Tanh()
    
    def forward(self, x):
        """Define forward pass"""
        x = self.activation(self.input_layer(x))  # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x))  # Apply hidden layer and ReLU
        return torch.tanh(self.output_layer(x))  # Return Q-values for all actions

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.input_layer_states = nn.Linear(state_size, 400)
        self.hidden_layer = nn.Linear(action_size+400, 200)
        self.output_layer = nn.Linear(200, 1)
        self.activation = nn.ReLU()
    
    def forward(self, state, action):
        """Define forward pass"""
        x = self.activation(self.input_layer_states(state))  # Apply input layer and 
        x = self.activation(self.hidden_layer(torch.concat((x,action), dim=1)))  # Apply hidden layer and ReLU
        return self.output_layer(x) # Return Q-values for all actions with duelling DQN

# initialize buffer
experience_buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
# initialise network
actor_network = ActorNetwork(input_size=env.observation_space.shape[0])
actor_network_target = ActorNetwork(input_size=env.observation_space.shape[0])

critic_network = CriticNetwork(state_size=env.observation_space.shape[0], action_size=m)
critic_network_target = CriticNetwork(state_size=env.observation_space.shape[0], action_size=m)

actor_network_target.load_state_dict(actor_network.state_dict())
critic_network_target.load_state_dict(critic_network.state_dict())


# optimizers
optimizer_critic = optim.Adam(critic_network.parameters(), lr=learning_rate_critic)
optimizer_actor = optim.Adam(actor_network.parameters(), lr=learning_rate_actor)


for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    noise = 0
    while not (done or truncated):
        # Take a random action
        action = actor_network(torch.tensor([state], dtype=torch.float32)).detach().numpy()[0]
        action += noise
        noise = compute_next_noise(noise)
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
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            # compute target values:
            with torch.no_grad():
                next_actions = actor_network_target(next_states)
                next_q_values = critic_network_target(next_states, next_actions).squeeze().detach()
                targets = rewards + discount_factor*next_q_values*(1-dones) # if terminal states then we don't change the Q value
            q_values = critic_network(states, actions).squeeze()
            loss_critic = nn.functional.mse_loss(q_values,targets)

            optimizer_critic.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=1.)
            optimizer_critic.step()

            if t % policy_update_freq == 0:
                # Initialize gradients
                actions_wrt_policy = actor_network(states)
                q_values_wrt_policy = critic_network(states, actions_wrt_policy)
                loss_actor = -torch.mean(q_values_wrt_policy)
                # Optimizer actor
                optimizer_actor.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=1.)
                optimizer_actor.step()
                # Update networks
                critic_network_target = soft_updates(critic_network, critic_network_target, 1E-3)
                actor_network_target = soft_updates(actor_network, actor_network_target, 1E-3)
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
    #break if early convergence
    if running_average(episode_reward_list, n_ep_running_average)[-1] > 150:
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

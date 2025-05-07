# Joachim JOBARD id:20020216-T352 & Farouk MILED id:20010524-T517
# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
# import torch
import matplotlib.pyplot as plt
import pickle

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 200        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
p = 2                   # fourier order               
l = 0.2
eps = 0.1
gamma = 0.9
learning_rate = 0.001

# init weights and momentum
momentum = 0.5


def alpha(eta, learning_rate):
    alphas = np.ones(eta.shape[0])*learning_rate
    for i in range(len(alphas)):
        current_norm = np.linalg.norm(eta[i])
        if current_norm!=0:
            alphas[i] /= np.linalg.norm(eta[i])
    return alphas


# Reward
  # Used to save episodes reward

def phi(s, eta):
    return np.cos(np.pi*eta@s)


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def Q_function(s,a, w_action, eta):
    # print(f'Q value: {w_action[a].T@phi(s, eta)}')
    return w_action[a].T@phi(s, eta)


# Training process
def train(l, eps, gamma, learning_rate, momentum):
    eta = np.array([[2, 1], [1, 1], [0, 1], [0, 0]])
    m = eta.shape[0]
    w_action = np.random.random((env.action_space.n, m))
    # w_action[:, 1] = -150
    # w_action = np.ones((env.action_space.n, m))*-10
    episode_reward_list = []
    alpha_ev = alpha(eta, learning_rate)
    for _ in range(N_episodes):
        # Reset enviroment data
        done = False
        truncated = False
        state = scale_state_variables(env.reset()[0])
        total_episode_reward = 0.
        v = np.zeros((env.action_space.n, m))

        # reset eligibility trace
        z = np.zeros((env.action_space.n, m))
        k = env.action_space.n
        if -50 > total_episode_reward > -150:
            alpha_ev*=0.70
        if np.random.uniform(0, 1) < eps:
            action = np.random.randint(0, k)
        else:
            action = np.argmax([Q_function(state, a, w_action, eta) for a in range(k)])

        while not (done or truncated):
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise. Truncated is true if you reach 
            # the maximal number of time steps, False else.
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state)
            if np.random.uniform(0, 1) < eps:
                next_action = np.random.randint(0, k)
            else:
                next_action = np.argmax([Q_function(next_state, a, w_action, eta) for a in range(k)])
            # Update episode reward
            total_episode_reward += reward

            # Update eligibility Trace
            for a in range(k):
                if a  == action:
                    z[a] = gamma*l*z[a] + phi(state, eta)
                else:
                    z[a] = gamma*l*z[a]

            # clipping
            z = np.clip(z, -5, 5)
            # Gradient descent
                # TD ERROR
            delta_t = reward + gamma*Q_function(next_state, next_action, w_action, eta) - Q_function(state, action, w_action, eta)
                # without optimisations
            # w_action += alpha*delta_t*z
                # Momentum with nesterov
            v = momentum * v + alpha_ev*delta_t*z
            w_action += momentum * v + alpha_ev*delta_t*z
            # Update state for next iteration
            state = next_state
            action = next_action

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()
        # Plot Rewards
    plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return np.mean(episode_reward_list), phi, w_action, eta


def grid_search():
    lambdas = np.linspace(0.1, 1, 5)
    momentums = np.linspace(0.1, 1, 5)
    gammas = np.linspace(0.1, 1, 5)
    learning_rates = np.linspace(0.1, 1, 5)
    epsilons = np.linspace(0.01, 0.4, 3)

    grid = -200*np.ones((len(lambdas), len(momentums), len(gammas), len(learning_rates), len(epsilons)))
    for i, lbd in enumerate(lambdas):
        for j, momentum in enumerate(momentums):
            for k, gamma in enumerate(gammas):
                for l, learning_rate in enumerate(learning_rates):
                    for e, epsilon in enumerate(epsilons):
                        result = train(lbd, epsilon, gamma, learning_rate, momentum)
                        grid[i,j,k,l,e] = result
            print(f'going through momentum={momentum}')
        print(f'going through lambda={lbd}')
    
    max_found = np.unravel_index(np.argmax(grid), grid.shape)
    print(max_found)
    print(f'max founded: lambda={lambdas[max_found[0]]}, momentum={momentums[max_found[1]]}, gammas={gammas[max_found[2]]}, learning rate={learning_rates[max_found[3]]}, epsilon={epsilons[max_found[4]]}')
    print(f'maximum: {np.max(grid)}')


mean, phi, w_action, eta = train(0.9, 0.0, 1, 0.001, 0.8)
with open('weights_marchent_pas.pkl', 'wb') as f:
    pickle.dump({'W':w_action, 'N':eta}, f)
f = open('weights_marchent_pas.pkl', 'rb')
data = pickle.load(f)
if 'W' not in data or 'N' not in data:
    print('Matrix W (or N) is missing in the dictionary.')
    exit(-1)
w = data['W']
eta = data['N']
print(w)


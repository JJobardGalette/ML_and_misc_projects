# Joachim JOBARD id:20020216-T352 & Farouk MILED id:20010524-T517
# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Modified version according to question h:
# - minotaur can stand still
# - life expectancy geometrically distributed with mean 50
# - minotaur can move toward the player with probability 35%
# - Now needing a key in order to open the labyrinth 

# This version will only work with Q-Learning so we would theoretically- 
# not need to compute the transition values.
# To translate the fact that we need a key, we will update :
# - Augment the states with a bollean telling if we have or not the key


from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
YELLOW       = '#FFFF66'

class MazeKey:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = -1          #put 1 and see what happens
    GOAL_REWARD = 100          #100 times higher.
    IMPOSSIBLE_REWARD = -2     #opposite to posible moves. Can be set lower
    MINOTAUR_REWARD = -100      #opposite to the goal reward
    KEY_REWARD = 0              #doesn't work
    


    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        # self.has_key                  = False
        self.maze                     = maze
        self.key_grabbed              = False
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        # self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()
        

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l), True)
                            map[((i,j), (k,l), True)] = s
                            s += 1
                            states[s] = ((i,j), (k,l), False)
                            map[((i,j), (k,l), False)] = s
                            s += 1 # added the key consideration
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map

    def move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y'), bool) on the maze that the system can transition to.
        """
        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return [self.states[state]]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = (row_player == -1) or \
                                             (row_player == self.maze.shape[0]) or \
                                             (col_player == -1) or \
                                             (col_player == self.maze.shape[1]) or\
                                             (self.maze[row_player,col_player] == 1)
        
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur, not allowed to stand still
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if self.states[state][0][0] == rows_minotaur[i] and self.states[state][0][1] == cols_minotaur[i]:                          # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]]==2 and self.states[state][2] is True:                           # TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if row_player == rows_minotaur[i] and col_player == cols_minotaur[i]:      # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif self.maze[row_player,col_player] == 2 and self.states[state][2]:       # TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    elif self.maze[row_player, col_player] == 3:
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), True))
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
              
                return states


    def __rewards(self):
        
        """ Computes the rewards for every state action pair """
        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                
                else:                
                    next_states = self.move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one

                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    elif self.maze[self.states[s][0][0], self.states[s][0][1]] == 3 and not self.states[s][2]:
                        rewards[s, a] = self.KEY_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards




    def simulate_Q_learning(self, start, Q, fixed_horizon=-1):
        path = list()
        t = 1 # Initialize current state, next state and time
        s = self.map[start]
        path.append(start) # Add the starting position in the maze to the path
        next_action = np.argmax(Q[s])
        next_states_map = self.move(s, next_action) # Move to next state given the policy and the current state
        if np.random.random() < 0.35: #Clever minotaur...
            if 'Eaten' in next_states_map:
                    next_state_map = 'Eaten'
            elif 'Win' not in next_states_map:
                distance_from_minotaur = [(s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2 for s in next_states_map]
                next_state_map = next_states_map[distance_from_minotaur.index(min(distance_from_minotaur))] #getting the state that minimize the distance function
            else:next_state_map = random.sample(next_states_map, 1)[0]
        else:
            next_state_map = random.sample(next_states_map, 1)[0] #minotaur moving randomly
        path.append(next_state_map) # Add the next state to the path
        # next_s = self.map[next_state_map]
        if fixed_horizon < 0:
            horizon = np.random.geometric(1/50)   
        else:
            horizon = fixed_horizon                          # Question e
        # Loop while state is not the goal state
        while (next_state_map != 'Eaten' or next_state_map != 'Win')  and t <= horizon:
            s = self.map[next_state_map] # Update state
            next_action = np.argmax(Q[s])
            next_states_map = self.move(s, next_action) # Move to next state given the policy and the current state
            if np.random.random() < 0.35: #Clever minotaur...
                if 'Eaten' in next_states_map:
                        next_state_map = 'Eaten'
                elif 'Win' not in next_states_map:
                    distance_from_minotaur = [(s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2 for s in next_states_map]
                    next_state_map = next_states_map[distance_from_minotaur.index(min(distance_from_minotaur))] #getting the state that minimize the distance function
                else:next_state_map = random.sample(next_states_map, 1)[0]
            else:
                next_state_map = random.sample(next_states_map, 1)[0] #minotaur moving randomly
            path.append(next_state_map) # Add the next state to the path
            next_s = next_state_map
            t += 1 # Update time for next iteration
    
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI


    def simulate_with_time(self, start, policy, method='ValIter'):
        path = list()    
        if method == 'ValIter': 
            p=1/30
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.move(s, policy[s,t-1]) # Move to next state given the policy and the current state
            next_s = next_states[np.random.randint(len(next_states))]
            path.append(next_s) # Add the next state to the path
            
            horizon = np.random.geometric(p)
            if horizon >= policy.shape[1]:
                horizon = policy.shape[1]-1                             # Question e
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = self.map[next_s] # Update state
                next_states = self.move(s, policy[s,t]) # Move to next state given the policy and the current state
                next_s = next_states[np.random.randint(len(next_states))]
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: YELLOW, -1: LIGHT_RED, -2: LIGHT_PURPLE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)

def q_learning(env:MazeKey, episodes, gamma=1, eps_decay=False, eps=0.1, alpha = 2/3, delta=0.6):
    # done taking DynaQ as an example
    start = ((0, 0), (0, 7), False)
    key_episode = env.map[((0, 7), (1, 6), True)]
    moving_episode = env.map[((4, 3), (4, 4), False)]
    start_states = []
    key_states = []
    moving_states = []
    V_values = []
    Q = defaultdict(lambda:np.zeros(5))# risks of having to change the action definition to an int... TBF
    # Need to initiate the Q value with the win reward for example and the key reward
    eaten_key = env.map['Eaten']
    win_key = env.map['Win']
    Q[eaten_key]+= env.MINOTAUR_REWARD
    Q[win_key]+= env.GOAL_REWARD

    number_of_visit = dict()
    for episode in range(episodes):
        # env.has_key = False #REALLY IMPORTANT!!!!
        if episode%100==0:
            delta = np.max(Q[env.map[start]])
            print(f'Going through episode {episode} out of episodes {episodes}, start V value={delta:.5f}',end='\r', flush=True)
            start_states.append(delta)
            key_states.append(np.max(Q[key_episode]))
            moving_states.append(np.max(Q[moving_episode]))
        if episode%1000==0:
            V_values.append(get_value(Q, env.n_states))
        state = env.map[start]
        done = False
        if eps_decay:
            eps = 1/((episode+1)**delta)
        while not done:
            actions = list(env.actions.keys())
            if np.random.uniform(0, 1) < eps:
                action = random.sample(actions, 1)[0]
            else:
                action = np.argmax(Q[state])
            next_states_map = env.move(state, action)
            if np.random.uniform(0, 1) < 0.35: #minotaur cleverness
                if 'Eaten' in next_states_map:
                    next_state_map = 'Eaten'
                elif 'Win' not in next_states_map:
                    distance_from_minotaur = [(s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2 for s in next_states_map]
                    next_state_map = next_states_map[distance_from_minotaur.index(min(distance_from_minotaur))] #getting the state that minimize the distance function
                else:next_state_map = random.sample(next_states_map, 1)[0]
            else:
                next_state_map = random.sample(next_states_map, 1)[0] #minotaur moving randomly
            next_state = env.map[next_state_map]
            reward = env.rewards[state, action]
            if (state,action) not in number_of_visit:
                number_of_visit[(state, action)] = 1
            else:
                number_of_visit[(state, action)] += 1
            
            # Q-learning
            best_next_action = np.argmax(Q[next_state])
            learning_rate = 1/((number_of_visit[(state, action)])**(alpha))
            Q[state][action] += learning_rate*(reward+gamma*Q[next_state][best_next_action]-Q[state][action])
            state = next_state
            if state == env.map['Eaten'] or state == env.map['Win']:
                done = True
    
    print('\n', flush=False)
    print('done')
    start_states.pop(0)
    key_states.pop(0)
    moving_states.pop(0)
    X = np.arange(100, episodes, 100)
    plt.plot(X, start_states, label='V value of starting state')
    plt.plot(X, key_states, label='V value of a key state')
    plt.plot(X, moving_states, label='V value of a moving to key state')
    plt.xlabel('Number of episodes')
    plt.ylabel('V function')
    plt.legend()
    plt.show()
    X_v = np.arange(1000, episodes, 1000)
    V_values.pop(0)
    plt.plot(X_v, V_values, color='red')
    plt.xlabel('Number of episodes')
    plt.ylabel('mean of V function over all the states')
    plt.plot
    return Q, number_of_visit

def SARSA(env:MazeKey, episodes, gamma=1, eps_decay=False, eps=0.1, alpha = 2/3, delta=0.6):
    # done taking DynaQ as an example
    start = ((0, 0), (0, 7), False)
    key_episode = env.map[((0, 7), (1, 6), True)]
    moving_episode = env.map[((4, 3), (4, 4), False)]
    start_states = []
    key_states = []
    moving_states = []
    V_values = []
    Q = defaultdict(lambda:np.zeros(5))# risks of having to change the action definition to an int... TBF
    # Need to initiate the Q value with the win reward for example and the key reward
    eaten_key = env.map['Eaten']
    win_key = env.map['Win']
    Q[eaten_key]+= env.MINOTAUR_REWARD
    Q[win_key]+= env.GOAL_REWARD

    number_of_visit = dict()
    actions = list(env.actions.keys())
    for episode in range(episodes):
        if episode%100==0:
            delta = np.max(Q[env.map[start]])
            print(f'Going through episode {episode} out of episodes {episodes}, start V value={delta:.5f}',end='\r', flush=True)
            start_states.append(delta)
            key_states.append(np.max(Q[key_episode]))
            moving_states.append(np.max(Q[moving_episode]))
        if episode%1000==0:
            V_values.append(get_value(Q, env.n_states))
        state = env.map[start]
        done = False
        if eps_decay:
            eps = 1/((episode+1)**delta)
        if np.random.uniform(0, 1) < eps:
            action = random.sample(actions, 1)[0]
        else:
            action = np.argmax(Q[state])
        while not done:
            next_states_map = env.move(state, action)
            if np.random.uniform(0, 1) < 0.35: #minotaur cleverness
                if 'Eaten' in next_states_map:
                    next_state_map = 'Eaten'
                elif 'Win' not in next_states_map:
                    distance_from_minotaur = [(s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2 for s in next_states_map]
                    next_state_map = next_states_map[distance_from_minotaur.index(min(distance_from_minotaur))] #getting the state that minimize the distance function
                else:next_state_map = random.sample(next_states_map, 1)[0]
            else:
                next_state_map = random.sample(next_states_map, 1)[0] #minotaur moving randomly
            next_state = env.map[next_state_map]
            reward = env.rewards[state, action]
            if (state,action) not in number_of_visit:
                number_of_visit[(state, action)] =1
            else:
                number_of_visit[(state, action)] +=1
            
            # Q-learning
            if np.random.uniform(0,1) < eps:
               next_action = random.sample(actions, 1)[0]
            else: 
                next_action = np.argmax(Q[next_state])
            learning_rate = 1/((number_of_visit[(state, action)])**(alpha))
            Q[state][action] += learning_rate*(reward+gamma*Q[next_state][next_action]-Q[state][action])
            state = next_state
            action = next_action
            if state == env.map['Eaten'] or state == env.map['Win']:
                done = True
    
    print('\n', flush=False)
    print('done')
    start_states.pop(0)
    key_states.pop(0)
    moving_states.pop(0)
    X = np.arange(100, episodes, 100)
    plt.plot(X, start_states, label='V value of starting state')
    plt.plot(X, key_states, label='V value of a key state')
    plt.plot(X, moving_states, label='V value of a moving to key state')
    plt.xlabel('Number of episodes')
    plt.ylabel('V function')
    plt.legend()
    plt.show()
    X_v = np.arange(1000, episodes, 1000)
    V_values.pop(0)
    plt.plot(X_v, V_values, color='red')
    plt.xlabel('Number of episodes')
    plt.ylabel('mean of V function over all the states')
    plt.plot
    return Q, number_of_visit

def q_learning_with_time(env:MazeKey, episodes, gamma=1, eps_decay=False, eps=0.1, alpha = 2/3):
    # done taking DynaQ as an example
    start = ((0, 0), (6, 5), False)
    key_episode = env.map[((0, 7), (1, 6), True)]
    moving_episode = env.map[((4, 3), (4, 4), False)]
    start_eps_decay = episodes
    start_states = []
    key_states = []
    moving_states = []
    V_values = []
    Q = defaultdict(lambda:np.zeros(5))# risks of having to change the action definition to an int... TBF
    # Need to initiate the Q value with the win reward for example and the key reward
    eaten_key = env.map['Eaten']
    win_key = env.map['Win']
    Q[eaten_key]+= env.MINOTAUR_REWARD
    Q[win_key]+= env.GOAL_REWARD

    stop_eps_decay = episodes//2
    eps_step = 1/(start_eps_decay-stop_eps_decay)
    if eps_decay:
        eps = 1
    number_of_visit = dict()
    for episode in range(episodes):
        if episode%100==0:
            delta = np.max(Q[env.map[start]])
            print(f'Going through episode {episode} out of episodes {episodes}, start V value={delta:.5f}',end='\r', flush=True)
            start_states.append(delta)
            key_states.append(np.max(Q[key_episode]))
            moving_states.append(np.max(Q[moving_episode]))
        if episode%1000==0:
            V_values.append(get_value(Q, env.n_states))
        state = env.map[start]
        done = False
        if eps_decay:
            if start_eps_decay >= episode >= stop_eps_decay:
                eps -= eps_step
        t = 0
        max_horizon = np.random.geometric(1/50)
        while not done or t <= max_horizon:
            t+=1
            actions = list(env.actions.keys())
            if np.random.uniform(0, 1) < eps:
                action = random.sample(actions, 1)[0]
            else:
                action = np.argmax(Q[state])
            next_states_map = env.move(state, action)
            if np.random.uniform(0, 1) < 0.35: #minotaur cleverness
                if 'Eaten' in next_states_map:
                    next_state_map = 'Eaten'
                elif 'Win' not in next_states_map:
                    distance_from_minotaur = [(s[0][0]-s[1][0])**2+(s[0][1]-s[1][1]) for s in next_states_map]
                    next_state_map = next_states_map[distance_from_minotaur.index(min(distance_from_minotaur))] #getting the state that minimize the distance function
                else:next_state_map = random.sample(next_states_map, 1)[0]
            else:
                next_state_map = random.sample(next_states_map, 1)[0] #minotaur moving randomly
            next_state = env.map[next_state_map]
            reward = env.rewards[state, action]
            if (state,action) not in number_of_visit:
                number_of_visit[(state, action)] =1
            else:
                number_of_visit[(state, action)] +=1
            
            # Q-learning
            best_next_action = np.argmax(Q[next_state])
            learning_rate = 1/((number_of_visit[(state, action)])**(alpha))
            Q[state][action] += learning_rate*(reward+gamma*Q[next_state][best_next_action]-Q[state][action])
            state = next_state
            if state == env.map['Eaten'] or state == env.map['Win']:
                done = True
    
    print('\n', flush=False)
    print('done')
    start_states.pop(0)
    key_states.pop(0)
    moving_states.pop(0)
    X = np.arange(100, episodes, 100)
    plt.plot(X, start_states, label='V value of starting state')
    plt.plot(X, key_states, label='V value of a key state')
    plt.plot(X, moving_states, label='V value of a moving to key state')
    plt.xlabel('Number of episodes')
    plt.ylabel('V function')
    plt.legend()
    plt.show()
    X_v = np.arange(1000, episodes, 1000)
    V_values.pop(0)
    plt.plot(X_v, V_values, color='red')
    plt.xlabel('Number of episodes')
    plt.ylabel('mean of V function over all the states')
    plt.plot
    return Q, number_of_visit

def get_value(Q,size):
    if size <= 0:
        return 0
    V = np.zeros(size)
    for i in range(len(V)):
        V[i] = np.max(Q[i])
    return np.mean(V)

if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze, 3 = key
    env = MazeKey(maze)
    Q,frequency_visited = q_learning(env, 50_000)
    start = ((0, 0), (6, 5), False)
    [path, horizon] = env.simulate_Q_learning(start, Q)

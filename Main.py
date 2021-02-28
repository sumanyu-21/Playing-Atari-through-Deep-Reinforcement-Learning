#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[44]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random

import gym

from DeepQNetwork import DeepQNetwork
from Agent import Agent

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# ## Utility Functions for calculating average reward and plotting graphs

# In[72]:


def calculate_avg_reward(scores, k):
    avg_scores = []
    running_sum = 0
    for i in range(k):
        running_sum+=scores[i]
        avg_scores.append(0)

    avg_scores.append(running_sum)

    for i in range(k,episodes,1):
        new_avg_score = avg_scores[-1] - scores[i-k] + scores[i]
        avg_scores.append(new_avg_score)
        
    avg_scores = np.array(avg_scores)/k
    return avg_scores



def plot_graphs(scores, avg_scores):
    plt.figure(1, figsize = (15,5))
    
    plt.subplot(121)
    plt.plot(scores)
    plt.xlabel('Current Episodes')
    plt.ylabel('Reward for each episode')
    plt.title('Rewards as we the agent learns ')
    plt.grid()
    
    plt.subplot(122)
    plt.plot(avg_scores)
    plt.xlabel('Current Episodes')
    plt.ylabel('Avg reward over a course of last 50 episodes')
    plt.title('Avg Rewards for past 50 episodes for a total of 500 episodes')
    plt.grid()


# ## Main Program
# 

# #### (i) Defining some variables

# In[45]:


env = gym.make('SpaceInvaders-v0')
num_actions = 6 # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
scores = []
episodes = 500
batch_size = 32


# #### (ii) Making an object of agent class and initialising Experience Replay memory with random transitions

# In[ ]:


agent = Agent(num_actions)


# In[46]:


while agent.memCntr < agent.memSize:
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done and info['ale.lives'] == 0: # TO avoid agent to loose, we are giving high penalty
            reward = -50
        agent.storeTransition(agent.process_state(state), action, reward, agent.process_state(next_state))
        state = next_state
print('done initializing memory')


# #### (iii) Main loop 

# In[10]:


for i in tqdm(range(episodes)):
    print('starting episode ', i+1, 'epsilon: %.4f' % agent.epsilon)
    done = False
    state = env.reset()
    frames = [agent.process_state(state)]
    score = 0
    lastAction = 0
    agent.updateEpsilon(i)
    while not done: # Action is repeated for 3 frames.
        if len(frames) == 3:
            action = agent.chooseAction(frames)
            frames = []
        else:
            action = lastAction
        next_state, reward, done, info = env.step(action)
        score += reward
        frames.append(agent.process_state(next_state))
        if done and info['ale.lives'] == 0:
            reward = -50
        agent.storeTransition(agent.process_state(state), action, reward, agent.process_state(next_state))
        state = next_state
        agent.learn(batch_size)
        lastAction = action
    scores.append(score)
    print('score:',score)


# In[60]:


k = 50
avg_scores = calculate_avg_reward(scores, k)


# #### Plotting graphs

# In[73]:


plot_graphs(scores, avg_scores)


# In[ ]:





# In[ ]:





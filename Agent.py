#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random

import gym
from DeepQNetwork import DeepQNetwork


# In[ ]:


class Agent(object):
    def __init__(self, num_actions, img_size = (1, 185, 95), gamma = .999, alpha = .003,
                maxMemorySize = 10_000, epsStart = 0.95, epsEnd = .05, decayRate = .05,
                replace = 10):
        
        self.gamma = gamma
        self.epsilon = epsStart
        self.epsStart = epsStart
        self.epsEnd = epsEnd
        self.decayRate = decayRate
        self.alpha = alpha
        
        self.num_actions = num_actions
        
        self.memSize = maxMemorySize
        self.memory = []
        self.memCntr = 0
        
        self.learn_counter = 0
        self.replace_target_cnt = replace
        
        self.Q_eval = DeepQNetwork(alpha, img_size, num_actions)
        self.Q_next = DeepQNetwork(alpha, img_size, num_actions)
     
    
    
    def storeTransition(self, state, action, reward, next_state):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, next_state])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward ,next_state]
        self.memCntr+=1
        
        
    def crop_image(self, state):
        return state[15:200, 30:125]

    
    def process_state(self, state):
        state = self.crop_image(state)
        return np.mean(state, axis = 2) 
    
    
    def updateEpsilon(self, current_episode):
        self.epsilon = self.epsEnd + (self.epsStart - self.epsEnd)*math.exp(-1*current_episode*self.decayRate)
     
   
    def chooseActionGreedy(self, observation):
        self.Q_eval.optimizer.zero_grad()
        actions = self.Q_eval.forward(observation)
        return torch.argmax(actions[1]).item()
    
    
    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1-self.epsilon:
            action = torch.argmax(actions[1]).item()
            
        else:
            action = np.random.choice(np.arange(self.num_actions))
        return action
    
    
    
    def learn(self,  batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.learn_counter%self.replace_target_cnt==0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            
        if batch_size < self.memCntr:
            miniBatch = random.sample(self.memory, batch_size)
            memory = np.array(miniBatch)
            
        # Converting to list to be given as input to the network
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device) # current_state prediction using Q_eval network 
        Qnext = self.Q_next.forward(list(memory[:,3][:])).to(self.Q_eval.device) # next_state prediction using Q_next network
        
        best_action = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = torch.Tensor(list(memory[:,2])).to(self.Q_eval.device)
        Qtarget = Qpred.clone() # We will only need to calculate loss for the best action and hence copying the entire tensor so that subtarction will result in zero loss across all other actions.
        indices = np.arange(batch_size)
        Qtarget[indices,best_action] = rewards + self.gamma*torch.max(Qnext[1])
        
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_counter += 1


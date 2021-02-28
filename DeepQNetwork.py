#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



# In[ ]:


class DeepQNetwork(nn.Module):
    def __init__(self, lr, img_size, num_actions):
        super(DeepQNetwork, self).__init__()
        self.img_size = img_size
        self.num_actions = num_actions
        self.lr = lr
        self.momentum = 0.9
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1)
        in_features = self.calculate_dim_fc_layer()
        self.fc1 = nn.Linear(in_features = in_features, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = self.num_actions) # There are 6 actions.
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = self.lr, momentum = self.momentum)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
        
        
        
    def calculate_dim_fc_layer(self):
        state = torch.zeros(1, *self.img_size)
        dims = self.conv1(state)
        return int(np.prod(dims.size()))
    
    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)                    # Observation is Heigh, Width , Channels
        observation =  observation.view(-1, 1, self.img_size[1], self.img_size[2]) # However we want channels to come first as we can see in convolution layer.
                                                                                   # -1 will take care of the number of frames we are passing.
        observation = F.relu(self.conv1(observation)) 
        observation = observation.flatten(start_dim = 1)
        observation = F.relu(self.fc1(observation))        
        actions = self.fc2(observation)
        return actions


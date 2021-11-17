import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        c, h, w = state_dim
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense = nn.Linear(64 * 7 * 7, 512)
        self.linear = nn.Linear(512, action_dim)
        
    def forward(self, state):
        state = state.__array__()
        state = torch.tensor(state)
        state = state.unsqueeze(0)

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.dense(state.reshape(-1, 64 * 7 * 7)))
        state = self.linear(state)

        return F.softmax(state, dim=1)
    
    def save_model(self):
        best_model_state = copy.deepcopy(self.state_dict())
        torch.save(best_model_state, "./trained_models/a2c_actor")

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        c, h, w = state_dim
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense = nn.Linear(64 * 7 * 7, 512)
        self.linear = nn.Linear(512, 1)
       
    def forward(self, state):
        state = state.__array__()
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.dense(state.reshape(-1, 64 * 7 * 7)))
        return self.linear(state)

    def save_model(self):
        best_model_state = copy.deepcopy(self.state_dict())
        torch.save(best_model_state, "./trained_models/a2c_actor")

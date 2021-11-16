import torch 
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_n, action_n):
        super(ActorCritic, self).__init__()

        c, h, w = input_n
        # Model layers (includes initialized model variables):
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(64 * 7 * 7, 512)
    
        #Activation function to get the estimate of value function
        #84 is the size of the input frame?/state?
        self.critic_linear = nn.Linear(512, 1)
        #Activation function to get policy distribution
        self.actor_linear = nn.Linear(512, action_n)

    def forward(self, state):
        state = state.__array__()
        state = torch.tensor(state)
        state = state.unsqueeze(0)

        state = F.relu((self.conv1(state)))
        state = F.relu((self.conv2(state)))
        state = F.relu((self.conv3(state)))
        state = F.relu(self.linear(state.reshape(-1, 64 * 7 * 7)))
    
        value = self.critic_linear(state)
        policy = F.softmax(self.actor_linear(state), dim=1)

        return value, policy
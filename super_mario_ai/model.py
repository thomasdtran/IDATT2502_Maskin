import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy

class ActorCritic(nn.Module):
    def __init__(self, input_n, action_n):
        super(ActorCritic, self).__init__()

        c, h, w = input_n
        # Model layers (includes initialized model variables):
        self.critic_conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.critic_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.critic_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.critic_dense = nn.Linear(64 * 7 * 7, 512)

        self.actor_conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.actor_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.actor_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.actor_dense = nn.Linear(64 * 7 * 7, 512)
    
        #Activation function to get the estimate of value functions
        #84 is the size of the input frame?/state?
        self.critic_linear = nn.Linear(512, 1)
        #Activation function to get policy distribution
        self.actor_linear = nn.Linear(512, action_n)

    def forward(self, state):
        state = state.__array__()
        state = torch.tensor(state)
        state = state.unsqueeze(0)

        value = F.relu((self.critic_conv1(state)))
        value = F.relu((self.critic_conv2(state)))
        value = F.relu((self.critic_conv3(state)))
        value = F.relu(self.critic_dense(state.reshape(-1, 64 * 7 * 7)))

        value = self.critic_linear(state)

        policy = F.relu((self.actor_conv1(state)))
        policy = F.relu((self.actor_conv2(state)))
        policy = F.relu((self.actor_conv3(state)))
        policy = F.relu(self.actor_dense(state.reshape(-1, 64 * 7 * 7)))

        policy = F.softmax(self.actor_linear(state), dim=1)

        return value, policy
    
    def save_model(self):
        best_model_state = copy.deepcopy(self.state_dict())
        torch.save(best_model_state, "./trained_models/a2c_super_mario")
    
    def save_checkpoint(self, optimizer):
        checkpoint_state = copy.deepcopy(self.state_dict())
        torch.save({
            'model_state_dict': checkpoint_state,
            'optimizer_state_dict': optimizer.state_dict(),
            }, "./model_checkpoints/a2c_checkpoint")

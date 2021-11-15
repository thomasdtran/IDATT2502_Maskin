import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_n, action_n):
        super(ActorCritic, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv1 = nn.Conv2d(input_n, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        #Activation function to get the estimate of value function
        #84 is the size of the input frame?/state?
        self.critic_linear = nn.Linear(32 * 84 * 84, 1)
        #Activation function to get policy distribution
        self.actor_linear = nn.Linear(32 * 84 * 84, action_n)

    def forward(self, state):

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.conv4(state))

        value = self.critic_linear(state)
        policy = F.softmax(self.actor_linear(state))

        return value, policy


"""def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))"""


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

env.close()


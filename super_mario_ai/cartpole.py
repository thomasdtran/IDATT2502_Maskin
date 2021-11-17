import sys
import matplotlib.pyplot as plt
import copy
import gym
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Actor(nn.Module):
    def __init__(self,action_dim, lr):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.dense = nn.Linear(864, 512)
        self.linear = nn.Linear(512, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.dense(state.reshape(-1, 864)))
        state = self.linear(state)

        return F.softmax(state, dim=1)

    def save_model(self):
        best_model_state = copy.deepcopy(self.state_dict())
        torch.save(best_model_state, "./trained_models/actor")


class Critic(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.dense = nn.Linear(864, 512)
        self.linear = nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = F.relu(self.dense(state.reshape(-1, 864)))
        return torch.tanh(self.linear(state))

    def save_model(self):
        best_model_state = copy.deepcopy(self.state_dict())
        torch.save(best_model_state, "./trained_models/critic")
  
    
class Cartpole():
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(40, interpolation=Image.CUBIC),
                                 T.ToTensor()])


    def get_cart_location(self,screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)

    def get_state(self):
        last_screen = self.get_screen()
        current_screen = self.get_screen()
        state = current_screen - last_screen
        return state


# hyperparameters
learning_rate = 1e-4
# Constants
GAMMA = 0.9
#num_steps = 3000
max_episodes = 5000


def a2c(env, cartpole):
    output_n = env.action_space.n

    actor = Actor(output_n, learning_rate)
    critic = Critic(learning_rate)

    all_rewards = []
    entropy_term = 0
    highest_reward = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []
        longest_run = 0

        env.reset()
        steps = 0

        state = cartpole.get_state()
        while True:
            env.render()
            steps += 1
            value = critic.forward(state)
            policy_dist = actor.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(output_n, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            _, reward, done, info = env.step(action)

            new_state = cartpole.get_state()
            #Just to keep track of how long the agent managed to run in each episode

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            state = new_state

            if done:
                Qval = critic.forward(new_state)
                Qval = Qval.detach().numpy()[0, 0]

                sum_rewards = np.sum(rewards)

                if(sum_rewards > highest_reward):
                    highest_reward = sum_rewards
                    actor.save_model()
                    critic.save_model()

                all_rewards.append(sum_rewards)

                sys.stdout.write("episode: {}, reward: {}, length: {} \n".format(
                    episode, np.sum(rewards), steps))

                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        critic.optimizer.zero_grad()
        actor.optimizer.zero_grad()
        ac_loss.backward()
        critic.optimizer.step()
        critic.optimizer.step()

        if((episode % 100 == 0) and (episode > 0)):
            torch.save({
                'epoch': episode,
                'critic_model_state_dict': copy.deepcopy(critic.state_dict()),
                'critic_optimizer_state_dict': critic.optimizer.state_dict(),
                'actor_model_state_dict': copy.deepcopy(actor.state_dict()),
                'actor_optimizer_state_dict': actor.optimizer.state_dict(),
            }, "model_checkpoints/checkpoint_{}".format(episode))

    env.close()
    data_interval = 20
    plot_reward(all_rewards, data_interval)


def plot_reward(all_rewards, interval):
    y = []
    x = []

    x_counter = 0
    for i in all_rewards[::interval]:
        y.append(i)
        x.append(x_counter)
        x_counter += interval

    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./plots/reward_episode")


cartpole = Cartpole()
env = cartpole.env
a2c(env, cartpole)

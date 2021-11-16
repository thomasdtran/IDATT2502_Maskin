import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from model import ActorCritic
from env import create_env



# hyperparameters
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 2000
max_episodes = 3000

def a2c(env, state_dim, action_dim):
    input_n = state_dim
    output_n = action_dim
    
    actor_critic = ActorCritic(input_n, output_n)
    ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    highest_reward = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []
        
        state = env.reset()
    
        for steps in range(num_steps):
            env.render()
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(output_n, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, info = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            state = new_state

            if done or steps == num_steps-1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]

                sum_rewards = np.sum(rewards)

                if(sum_rewards > highest_reward):
                    highest_reward = sum_rewards
                    best_model_state = copy.deepcopy(actor_critic.state_dict())
                    torch.save(best_model_state, "./trained_models/a2c_super_mario")

                all_rewards.append(sum_rewards)
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                sys.stdout.write("episode: {}, reward: {}, steps: {}, traversed_length: {} \n".format(episode, np.sum(rewards), steps, info["x_pos"]))

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

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
    
    env.close()
    draw_plot(all_rewards)

def draw_plot(all_rewards):
    y = []
    x = []

    x_counter = 0
    for i in all_rewards[::20]:
        x_counter += 20
        y.append(i)
        x.append(x_counter)

    plt.plot(x,y)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./plots/reward_episode")


env, state_dim, action_dim = create_env()

a2c(env, state_dim, action_dim) 


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
learning_rate = 1e-4
# Constants
GAMMA = 0.9
#num_steps = 3000
max_episodes = 5000

def a2c(env, state_dim, action_dim):
    input_n = state_dim
    output_n = action_dim
    
    actor_critic = ActorCritic(input_n, output_n)
    actor_critic.load_state_dict(torch.load("./trained_models/a2c_super_mario"))
    ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_rewards = []
    entropy_term = 0
    highest_reward = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []
        longest_run = 0
        
        state = env.reset()
        steps = 0

        while True:
            env.render()
            steps +=1 
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(output_n, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, info = env.step(action)

            #Just to keep track of how long the agent managed to run in each episode
            run_length = info["x_pos"]
            if(run_length > longest_run):
                longest_run = run_length

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            state = new_state

            if done:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]

                sum_rewards = np.sum(rewards)

                if(sum_rewards > highest_reward):
                    highest_reward = sum_rewards
                    actor_critic.save_model()

                all_rewards.append(sum_rewards)

                if(episode % 50 == 0 and episode > 0):
                    actor_critic.save_checkpoint(ac_optimizer)

                sys.stdout.write("episode: {}, reward: {}, length: {}, longest traversel: {} \n".format(episode, np.sum(rewards), steps, longest_run))

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

    plt.plot(x,y)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./plots/reward_episode")

env, state_dim, action_dim = create_env()

a2c(env, state_dim, action_dim) 


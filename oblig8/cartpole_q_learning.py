import numpy as np
import random
import math
import gym

env = gym.make('CartPole-v0')

buckets = (1, 1, 6, 12)

# define upper and lower bounds for each state value
upper_bounds = [
    env.observation_space.high[0],
    0.5,
    env.observation_space.high[2],
    math.radians(50)
]
lower_bounds = [
    env.observation_space.low[0],
    -0.5,
    env.observation_space.low[2],
    -math.radians(50)]

Q = np.zeros(buckets + (env.action_space.n,))

# Hyperparameters
gamma = 1
min_alpha = 0.1
min_epsilon = 0.1
n_episodes = 1000
n_steps = 200
ada_divisor = 25  # decay rate parameter for alpha and epsilon


def discretize(obs):
    ''' discretise the continuous state into buckets '''
    ratios = [(obs[i] + abs(lower_bounds[i])) /
              (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i]))
               for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i]))
               for i in range(len(obs))]
    return tuple(new_obs)


def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))


def get_alpha(t):
    ''' decrease the learning rate at each episode '''
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


rewards = []

for episode in range(n_episodes):

    state = discretize(env.reset())

    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    episode_rewards = 0

    for t in range(n_steps):
        env.render()
        if(random.uniform(0, 1) < epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        new_state, reward, done, info = env.step(action)

        new_state = discretize(new_state)

        ''' update the Q matrix with the Bellman equation '''
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        state = new_state

        episode_rewards += reward

        if done:
            print("Episode{}/{} ended with a total reward of {}".format(episode,
                  n_episodes, episode_rewards))
            break

    rewards.append(episode_rewards)

env.close()

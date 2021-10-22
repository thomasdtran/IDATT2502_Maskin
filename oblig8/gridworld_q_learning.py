from numpy.core.fromnumeric import argmax
import pygame
import random
import numpy as np
import math

class Environment:
    def __init__(self, blockSize, window_height, window_width):
        #5x5 grid, where the agent starts at the bottom left corner of the grid
        self.currentState = [0, ((window_height / blockSize) - 1) * blockSize]
        self.blockSize =blockSize
        self.window_height = window_height
        self.window_width = window_width
        self.allStates = []
        self.WIN_STATE = [((window_width / blockSize) - 1) * blockSize, 0] #Upper right corner 
        self.DANGER_STATE = [[0,0],
                             [((window_width / blockSize) - 1) * blockSize, ((window_height / blockSize) - 1) * blockSize]]
        self.action = [0,1,2,3] #Available actions
        self.isDone = False
        self.reward = 0
    
    def setup(self):
        for x in range(0, self.window_height, self.blockSize):
            for y in range(0, self.window_width, self.blockSize):
                state = [x, y]
                self.allStates.append(state)

    def reset(self):
        self.currentState = [0, ((self.window_height / self.blockSize) - 1) * self.blockSize]
        self.isDone = False
        self.reward = 0
        return self.currentState
    
    def step(self, action):
        """
        Different types of actions represented with numbers
        0 = up
        1 = down
        2 = left
        3 = rigth

        Thinking about the x-axis and y-axis in pygame.
        Translated to the matrix: [x, y]
        """
        
        if action == 0:
            nextState = [0, -1]
        elif action == 1:
            nextState = [0, 1]
        elif action == 2:
            nextState = [-1, 0]
        elif action == 3:
            nextState = [1, 0]

        stepSize = self.blockSize
        nextState = [self.currentState[0] + nextState[0] * stepSize, self.currentState[1] + nextState[1] * stepSize]

        if(nextState[0] >= 0) and (nextState[0] <= ((self.window_width / self.blockSize) - 1) * self.blockSize):
            if(nextState[1] >= 0) and (nextState[1] <= ((self.window_height / self.blockSize)- 1) * self.blockSize):
                if nextState == self.WIN_STATE:
                    self.reward = 1
                    self.isDone = True
                elif (nextState == self.DANGER_STATE[0]) or (nextState == self.DANGER_STATE[1]):
                    self.reward = -1
                self.currentState = nextState
        else:
            self.reward = -1
        return self.currentState


# Hyperparameters
gamma = 1
min_alpha = 0.1
min_epsilon = 0.1
n_episodes = 1000
n_steps = 200
ada_divisor = 25  # decay rate parameter for alpha and epsilon

def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))


def get_alpha(t):
    ''' decrease the learning rate at each episode '''
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


#Constants for the pygame visualization
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)  # Goal
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400
FPS = 60
blockSize = 80  # Set the size of the grid block

env = Environment(blockSize, WINDOW_HEIGHT, WINDOW_WIDTH)
env.setup()

#Our Q-list to keep track of the best possible action to take given a state
Q = np.zeros((len(env.allStates), (len(env.action))))

#Q-learning
for episode in range(n_episodes):
    env.reset()
    state =  env.allStates.index(env.currentState)

    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    episode_score = 0

    for t in range(n_steps):
        if(random.uniform(0, 1) < epsilon):
            action = random.randrange(len(env.action))
        else:
            action = np.argmax(Q[state])

        new_state = env.allStates.index(env.step(action))
        reward = env.reward
        done = env.isDone

        ''' update the Q matrix with the Bellman equation '''
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        episode_score += reward
        state = new_state

        if done:
            print("Episode{}/{} ended after a score of: {}".format(episode,
                  n_episodes, episode_score))
            break

#Drawing the gridworld as well as the best choices for each state
def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    running = True
    while running:
        CLOCK.tick(FPS)
        drawGrid()
        drawQ()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.update()
    
    pygame.quit()

def drawGrid():
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(0, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)

    goal = pygame.Rect(4*80, 0, blockSize, blockSize)
    pygame.draw.rect(SCREEN, GREEN, goal)

def drawQ():
    for state in range(len(Q)):
        if env.allStates[state] != env.WIN_STATE:
            best_action = np.argmax(Q[state])
            xPos = env.allStates[state][0]
            yPos = env.allStates[state][1]
            rect = pygame.Rect(xPos, yPos, blockSize - 20, blockSize - 20)
            """
            0 = Up = blue
            1 = down = orange
            2 = left = purple
            3 = right = yellow
            """
            if(best_action == 0):
                color = (0, 0, 255)
            elif(best_action == 1):
                color = (255, 153, 51)
            elif(best_action == 2):
                color = (102, 0, 204)
            elif(best_action == 3):
                color = (255, 255, 0)
            pygame.draw.rect(SCREEN, color, rect)


main()

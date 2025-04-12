import sys
sys.path.append("..")
from collections import defaultdict, deque
from common.gridworld import GridWorld
from common.utils import greedy_probs
import numpy as np


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {action: 1.0 / self.action_size for action in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax([self.Q[(state, a)] for a in range(self.action_size)])

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_q_max = max(self.Q[(next_state, a)] for a in range(self.action_size))

        td_target = reward + self.gamma * next_q_max
        self.Q[(state, action)] += self.alpha * (td_target - self.Q[(state, action)])

    def get_policy(self):
        for key in self.Q.keys():
            state = key[0]
            action_values = [self.Q[(state, a)] for a in range(self.action_size)]
            best_action = np.argmax(action_values)
            self.pi[state] = {a: 0.0 for a in range(self.action_size)}
            self.pi[state][best_action] = 1.0


env = GridWorld()
agent = QLearningAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(state, action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break

        state = next_state

agent.get_policy()
env.render_q(agent.Q, agent.pi)

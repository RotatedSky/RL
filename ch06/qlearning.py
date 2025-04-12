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
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_q_max = max(self.Q[(next_state, a)] for a in range(self.action_size))

        td_target = reward + self.gamma * next_q_max
        self.Q[(state, action)] += self.alpha * (td_target - self.Q[(state, action)])

        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


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

env.render_q(agent.Q, agent.pi)

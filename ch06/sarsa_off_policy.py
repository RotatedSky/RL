import sys
sys.path.append("..")
from collections import defaultdict, deque
from common.gridworld import GridWorld
from common.utils import greedy_probs
import numpy as np


class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {action: 1.0 / self.action_size for action in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()
    
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1.0
        else:
            next_q = self.Q[(next_state, next_action)]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        td_target = rho * (reward + self.gamma * next_q)
        self.Q[(state, action)] += self.alpha * (td_target - self.Q[(state, action)])

        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaOffPolicyAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(state, action)

        agent.update(state, action, reward, done)

        if done:
            agent.update(state, action, reward, done)
            break

        state = next_state

env.render_q(agent.Q, agent.pi)

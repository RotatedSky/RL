import sys
sys.path.append("..")
from collections import defaultdict
from common.gridworld import GridWorld
from policy_eval import policy_eval
from policy_iter import policy_iter, greedy_policy
from value_iter import value_iter

env = GridWorld()
gamma = 0.9

## method 1
# pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
# V = defaultdict(lambda: 0)
# V = policy_eval(pi, V, env, gamma)
# env.render_v(V, pi)

## method 2
# pi = policy_iter(env, gamma, 1e-3, True)

## method 3
V = defaultdict(lambda: 0)
V = value_iter(V, env, gamma)
pi = greedy_policy(V, env, gamma)
env.render_v(V, pi)

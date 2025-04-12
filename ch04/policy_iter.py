from collections import defaultdict
from policy_eval import policy_eval

def argmax(d):
    max_value = max(d.values())
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma):
    pi = {}
    for state in env.states:
        action_values = {}
        for action in env.actions:
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]
        max_action = argmax(action_values)
        action_probs = { 0: 0, 1: 0, 2: 0, 3: 0 }
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=1e-3, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        if new_pi == pi:
            break
        pi = new_pi
        if is_render:
            env.render_v(V, pi)
    return pi

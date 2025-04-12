
def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states:
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=1e-3):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V

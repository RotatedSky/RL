
def value_iter_onestep(V, env, gamma):
    for state in env.states:
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions:
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=1e-3):
    while True:
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V

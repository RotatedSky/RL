import numpy as np

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right"
        }
        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return self.reward_map.shape[0]
    
    @property
    def width(self):
        return self.reward_map.shape[1]
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    @property
    def actions(self):
        return self.action_space
    
    @property
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if ny < 0 or ny >= self.height or nx < 0 or nx >= self.width:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    def step(self, state, action):
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state
        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, V, pi):
        for h in range(self.height):
            for w in range(self.width):
                state = (h, w)
                if state == self.goal_state:
                    print(f"[{state}, Goal:{V[state]}, pi:{pi[state]}]")
                elif state == self.wall_state:
                    print(f"[{state}, Wall:{V[state]}, pi:{pi[state]}]")
                else:
                    print(f"[{state}, {V[state]}, pi:{pi[state]}]")

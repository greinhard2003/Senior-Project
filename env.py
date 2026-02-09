import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

from cube import Cube, moves

def cube_distance(state: Cube) -> int:
    misplaced_corner = sum(1 for i in range(8) if state.cp[i] != i)
    misplaced_edge   = sum(1 for i in range(12) if state.ep[i] != i)
    twisted_corner   = sum(1 for i in range(8)  if state.cp[i] == i and state.co[i] != 0)
    flipped_edge     = sum(1 for i in range(12) if state.ep[i] == i and state.eo[i] != 0)
    return 3 * misplaced_corner + 2 * misplaced_edge + twisted_corner + flipped_edge

def get_reward(d0: int, d1: int, solved: bool) -> float:
    max_distance = 68
    reward = (d0 - d1) / max_distance - 0.01
    if solved:
        reward += 10.0
    return reward

def encode_state(state: Cube) -> np.ndarray:
    x = np.array(state.cp + state.co + state.ep + state.eo, dtype=np.float32)
    x[0:8]  /= 7.0
    x[8:16] /= 2.0
    x[16:28] /= 11.0
    # eo already 0/1
    return x

class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=5, max_steps=200):
        super().__init__()
        self.scramble_len = scramble_len
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(len(moves))  # 12
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(40,), dtype=np.float32
        )

        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.state = Cube()
        for _ in range(self.scramble_len):
            a = random.randrange(len(moves))
            self.state = self.state.apply_move(moves[a])

        self.steps = 0
        obs = encode_state(self.state)
        info = {"distance": cube_distance(self.state)}
        return obs, info

    def step(self, action):
        d0 = cube_distance(self.state)
        self.state = self.state.apply_move(moves[int(action)])
        d1 = cube_distance(self.state)

        self.steps += 1
        solved = self.state.is_solved()

        reward = get_reward(d0, d1, solved)
        obs = encode_state(self.state)

        terminated = solved
        truncated = (self.steps >= self.max_steps)
        info = {"distance": d1, "solved": solved, "steps": self.steps}

        return obs, reward, terminated, truncated, info

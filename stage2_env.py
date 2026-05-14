import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import copy
import pickle

from cube import Cube, moves

# -------------------------
# Distance functions
# -------------------------

def stage1_distance(c: Cube):
    return (
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [1, 5, 8, 9]) +
        2 * sum(int(c.cp[i] != i) + int(c.co[i] != 0) for i in [0, 1, 4, 5])
    )

def stage2_distance(c: Cube):
    return (
        stage1_distance(c) +
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [0, 2, 4, 6])
    )

def is_stage2_complete(c: Cube):
    return stage2_distance(c) == 0

# -------------------------
# Reward
# -------------------------

def stage2_reward(d0, d1, done, stage1_broken):
    delta = d0 - d1

    r = delta * 0.7

    if delta > 0:
        r += 0.2
    elif delta < 0:
        r -= 0.3

    r -= 0.02

    if stage1_broken:
        r -= 1.0
    if done:
        r += 8.0

    return r



class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=10, max_steps=150):
        super().__init__()

        self.stage1_buffer = []
        self.used_buffer = False
        self.stage2_buffer = []
        self.max_buffer_size = 10000

        self.scramble_min = 1
        self.scramble_max = scramble_len
        self.max_steps = int(max_steps)

        self.action_space = spaces.Discrete(len(moves))

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(40,), dtype=np.float32
        )

        self.state: Cube | None = None
        self.steps = 0

    def load_stage1_buffer(self, path="stage1_buffer.pkl"):
        with open(path, "rb") as f:
            self.stage1_buffer = pickle.load(f)

    def get_obs(self):
        cp = np.array(self.state.cp) / 7.0
        co = np.array(self.state.co) / 2.0
        ep = np.array(self.state.ep) / 11.0
        eo = np.array(self.state.eo)
        return np.concatenate([cp, co, ep, eo]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        s = random.choice(self.stage1_buffer)
        self.state = copy.deepcopy(s)

        self.steps = 0

        return self.get_obs(), {
            "distance": stage2_distance(self.state),
        }

    def step(self, action):
        assert self.state is not None

        action = int(action)

        d0 = stage2_distance(self.state)

        self.state = self.state.apply_move(moves[action])

        d1 = stage2_distance(self.state)

        self.steps += 1

        done = is_stage2_complete(self.state)

        stage1_broken = stage1_distance(self.state) > 0

        if done:
            if len(self.stage2_buffer) < self.max_buffer_size and random.random() < 0.3:
                self.stage2_buffer.append(copy.deepcopy(self.state))

        reward = stage2_reward(d0, d1, done, stage1_broken)

        terminated = done
        truncated = self.steps >= self.max_steps

        return self.get_obs(), reward, terminated, truncated, {
            "distance": d1,
            "solved_stage2": done,
            "steps": self.steps,
        }
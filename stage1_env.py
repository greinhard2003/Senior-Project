import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import torch
import copy
import pickle

from cube import Cube, moves

def stage1_distance(c: Cube):
    return (
        # F edges (cross)
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [1, 5, 8, 9]) +
        # F corners (weighted higher)
        2 * sum(int(c.cp[i] != i) + int(c.co[i] != 0) for i in [0, 1, 4, 5])
    )

def is_stage1_complete(c: Cube):
    return stage1_distance(c) == 0


def stage1_reward(d0, d1, done):
    delta = d0 - d1

    r = delta * 0.8   # slightly lower than stage0 (harder task, smoother gradients)

    if delta > 0:
        r += 0.2
    elif delta < 0:
        r -= 0.3

    r -= 0.02  # slightly higher step penalty

    if done:
        r += 6.0  # stronger completion reward

    return r

class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=8, max_steps=100):
        super().__init__()

        self.stage0_buffer = []

        self.stage1_buffer = []
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

    def get_obs(self):
        cp = np.array(self.state.cp) / 7.0
        co = np.array(self.state.co) / 2.0
        ep = np.array(self.state.ep) / 11.0
        eo = np.array(self.state.eo)
        return np.concatenate([cp, co, ep, eo]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = copy.deepcopy(random.choice(self.stage0_buffer))

        self.steps = 0

        return self.get_obs(), {
            "distance": stage1_distance(self.state),
        }

    def step(self, action):
        assert self.state is not None

        action = int(action)

        d0 = stage1_distance(self.state)

        self.state = self.state.apply_move(moves[action])

        d1 = stage1_distance(self.state)

        self.steps += 1

        done = is_stage1_complete(self.state)

        reward = stage1_reward(d0, d1, done)

        cross_broken = sum(
            int(self.state.ep[i] != i) + int(self.state.eo[i] != 0)
            for i in [1, 5, 8, 9]
        ) > 0

        if cross_broken:
            reward -= 1.0

        if done:
            if len(self.stage1_buffer) < self.max_buffer_size and random.random() < 0.3:
                self.stage1_buffer.append(copy.deepcopy(self.state))

        terminated = done
        truncated = self.steps >= self.max_steps

        return self.get_obs(), reward, terminated, truncated, {
            "distance": d1,
            "solved_stage1": done,
            "steps": self.steps,
        }

    def load_stage0_buffer(self, path="stage0_buffer.pkl"):
        with open(path, "rb") as f:
            self.stage0_buffer = pickle.load(f)
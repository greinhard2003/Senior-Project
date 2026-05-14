import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import torch
import copy
import pickle

from cube import Cube, moves

def stage0_distance(c: Cube):
    return sum(
        int(c.ep[i] != i) + int(c.eo[i] != 0)
        for i in [1, 5, 8, 9]
    )

def is_stage0_complete(c: Cube):
    return stage0_distance(c) == 0


def stage0_reward(d0, d1, done):
    delta = d0 - d1

    r = delta * 1.0

    if delta > 0:
        r += 0.3
    elif delta < 0:
        r -= 0.3

    r -= 0.01

    if done:
        r += 5.0

    return r
class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=8, max_steps=80):
        super().__init__()
        self.stage0_buffer = []
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

        scramble_len = 25

        self.state = Cube()
        for _ in range(scramble_len):
            a = int(self.np_random.integers(0, len(moves)))
            self.state = self.state.apply_move(moves[a])

        self.steps = 0

        return self.get_obs(), {
            "scramble_len": scramble_len,
            "distance": stage0_distance(self.state),
        }

    def step(self, action):
        assert self.state is not None

        action = int(action)

        d0 = stage0_distance(self.state)

        self.state = self.state.apply_move(moves[action])

        d1 = stage0_distance(self.state)

        self.steps += 1

        done = is_stage0_complete(self.state)

        if done:
            if len(self.stage0_buffer) < self.max_buffer_size and random.random() < 0.3:
                self.stage0_buffer.append(copy.deepcopy(self.state))

        reward = stage0_reward(d0, d1, done)

        terminated = done
        truncated = self.steps >= self.max_steps

        return self.get_obs(), reward, terminated, truncated, {
            "distance": d1,
            "solved_cross": done,
            "steps": self.steps,
        }

    def save_stage0_buffer(self, path="stage0_buffer.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.stage0_buffer, f)

    def load_stage0_buffer(self, path="stage0_buffer.pkl"):
        with open(path, "rb") as f:
            self.stage0_buffer = pickle.load(f)
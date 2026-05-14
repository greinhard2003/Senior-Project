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

def stage2_distance(c: Cube):
    return (
        # F edges
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [1,5,8,9]) +
        # F corners
        2 * sum(int(c.cp[i] != i) + int(c.co[i] != 0) for i in [0,1,4,5]) +
        # middle edges
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [0,2,4,6])
    )

def stage3_distance(c: Cube):
    return (
        stage2_distance(c) +
        # last layer edges (orientation + permutation)
        sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [3,7,10,11]) +
        # last layer corners
        2 * sum(int(c.cp[i] != i) + int(c.co[i] != 0) for i in [2,3,6,7])
    )

def is_stage3_complete(c: Cube):
    return c.is_solved()


def stage3_reward(d0, d1, done, broke_stage2):
    delta = d0 - d1

    r = delta * 0.6  # smaller signal (harder task)

    if delta > 0:
        r += 0.2
    elif delta < 0:
        r -= 0.4

    r -= 0.03

    if broke_stage2:
        r -= 2

    if done:
        r += 15.0

    return r


class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=12, max_steps=200):
        super().__init__()

        self.stage2_buffer = []
        self.scramble_min = 1
        self.scramble_max = scramble_len
        self.max_steps = int(max_steps)

        self.action_space = spaces.Discrete(len(moves))

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(40,), dtype=np.float32
        )

        self.state: Cube | None = None
        self.steps = 0

        # 🔥 buffer from stage 2
        self.stage2_buffer = []


    def load_stage2_buffer(self, path="stage2_buffer.pkl"):
        with open(path, "rb") as f:
            self.stage2_buffer = pickle.load(f)


    def get_obs(self):
        cp = np.array(self.state.cp) / 7.0
        co = np.array(self.state.co) / 2.0
        ep = np.array(self.state.ep) / 11.0
        eo = np.array(self.state.eo)
        return np.concatenate([cp, co, ep, eo]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        s = random.choice(self.stage2_buffer)

        self.state = copy.deepcopy(s)


        self.steps = 0

        return self.get_obs(), {
            "distance": stage3_distance(self.state),
        }

    def step(self, action):
        assert self.state is not None

        action = int(action)

        d0 = stage3_distance(self.state)

        self.state = self.state.apply_move(moves[action])

        d1 = stage3_distance(self.state)

        self.steps += 1

        done = is_stage3_complete(self.state)

        # check if stage2 got broken
        broke_stage2 = stage2_distance(self.state) > 0

        reward = stage3_reward(d0, d1, done, broke_stage2)

        terminated = done
        truncated = self.steps >= self.max_steps

        return self.get_obs(), reward, terminated, truncated, {
            "distance": d1,
            "solved": done,
            "steps": self.steps,
        }
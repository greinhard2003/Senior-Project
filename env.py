import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

from cube import Cube, moves

def stage_distance(c, stage):
    if stage == 0:
        # count wrong D edges (position or flip)
        return sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [4,5,6,7])
    if stage == 1:
        return (
            sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [4,5,6,7]) +
            2*sum(int(c.cp[i] != i) + int(c.co[i] != 0) for i in [4,5,6,7])
        )
    if stage == 2:
        return (
            stage_distance(c, 1) +
            sum(int(c.ep[i] != i) + int(c.eo[i] != 0) for i in [8,9,10,11])
        )
    if stage == 3:
        return (
            stage_distance(c, 2) +
            sum(int(c.eo[i] != 0) for i in [0,1,2,3]) +
            sum(int(c.co[i] != 0) for i in [0,1,2,3])
        )
    return 0 if c.is_solved() else 1


def stage_reward(d0, d1, stage_completed, broke_previous):
    r = (d0 - d1) * 0.2 - 0.01
    if stage_completed:
        r += 2.0
    if broke_previous:
        r -= 1.0
    return r

def is_stage_complete(c: Cube, stage: int) -> bool:
    # “complete” means distance == 0 for that stage’s definition
    if stage <= 3:
        return stage_distance(c, stage) == 0
    return c.is_solved()

def encode_state(state: Cube, stage: int) -> np.ndarray:
    x = np.array(state.cp + state.co + state.ep + state.eo, dtype=np.float32)
    x[0:8] /= 7.0
    x[8:16] /= 2.0
    x[16:28] /= 11.0


    stage_onehot = np.zeros(5, dtype=np.float32)
    stage_onehot[int(stage)] = 1.0

    return np.concatenate([x, stage_onehot], axis=0)

class CubeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scramble_len=5, max_steps=200):
        super().__init__()
        self.scramble_min = 1
        self.scramble_max = scramble_len  # initial difficulty cap

        self.max_steps = int(max_steps)

        self.action_space = spaces.Discrete(len(moves))  # 12

        # 40 state + 5 stage one-hot = 45
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(45,), dtype=np.float32
        )

        self.state: Cube | None = None
        self.steps = 0
        self.stage = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        scramble_len = int(self.np_random.integers(self.scramble_min, self.scramble_max + 1))
        self.state = Cube()
        for _ in range(scramble_len):
            a = int(self.np_random.integers(0, len(moves)))
            self.state = self.state.apply_move(moves[a])

        self.steps = 0
        self.stage = 0

        obs = encode_state(self.state, self.stage)
        info = {
            "scramble_len": scramble_len,
            "stage": self.stage,
            "stage_distance": stage_distance(self.state, self.stage),
        }
        return obs, info

    def step(self, action):
        assert self.state is not None, "Call reset() before step()."

        action = int(action)

        # Measure distance for current stage before/after
        d0 = stage_distance(self.state, self.stage)
        self.state = self.state.apply_move(moves[action])
        d1 = stage_distance(self.state, self.stage)

        self.steps += 1

        # Did we break a previous stage? (regression penalty)
        broke_previous = any(stage_distance(self.state, s) != 0 for s in range(self.stage))

        # Stage completion + advance
        stage_completed = is_stage_complete(self.state, self.stage)
        if stage_completed and self.stage < 4:
            self.stage += 1  # advance to next stage

        solved = self.state.is_solved()

        reward = stage_reward(d0, d1, stage_completed, broke_previous)
        obs = encode_state(self.state, self.stage)

        terminated = solved
        truncated = (self.steps >= self.max_steps)

        info = {
            "stage": self.stage,
            "stage_completed": stage_completed,
            "broke_previous": broke_previous,
            "stage_distance": stage_distance(self.state, self.stage) if self.stage <= 4 else 0,
            "solved": solved,
            "steps": self.steps,
        }

        return obs, reward, terminated, truncated, info
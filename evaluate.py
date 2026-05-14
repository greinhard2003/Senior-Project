import numpy as np
import random
from stable_baselines3 import PPO

from cube import Cube, moves


stage0_model = PPO.load("multi_models/stage0")
stage1_model = PPO.load("multi_models/stage1")
stage2_model = PPO.load("multi_models/stage2")
stage3_model = PPO.load("multi_models/stage3")


def get_obs(cube: Cube):
    cp = np.array(cube.cp) / 7.0
    co = np.array(cube.co) / 2.0
    ep = np.array(cube.ep) / 11.0
    eo = np.array(cube.eo)
    return np.concatenate([cp, co, ep, eo]).astype(np.float32)



def stage0_done(c):
    return all(c.ep[i] == i and c.eo[i] == 0 for i in [1,5,8,9])

def stage1_done(c):
    return stage0_done(c) and all(
        (c.cp[i] == i and c.co[i] == 0) for i in [0,1,4,5]
    )

def stage2_done(c):
    return stage1_done(c) and all(
        (c.ep[i] == i and c.eo[i] == 0) for i in [0,2,4,6]
    )

def stage3_done(c):
    return c.is_solved()


def run_stage(model, cube, done_fn, max_steps=100):
    for _ in range(max_steps):
        obs = get_obs(cube)
        action, _ = model.predict(obs, deterministic=True)
        cube = cube.apply_move(moves[int(action)])

        if done_fn(cube):
            return cube, True

    return cube, False


def solve_cube(scramble_len=20):
    cube = Cube()

    # scramble
    for _ in range(scramble_len):
        cube = cube.apply_move(random.choice(moves))

    # Stage 0
    cube, ok = run_stage(stage0_model, cube, stage0_done)
    if not ok:
        return False

    # Stage 1
    cube, ok = run_stage(stage1_model, cube, stage1_done)
    if not ok:
        return False

    # Stage 2
    cube, ok = run_stage(stage2_model, cube, stage2_done)
    if not ok:
        return False

    # Stage 3
    cube, ok = run_stage(stage3_model, cube, stage3_done)
    if not ok:
        return False

    return cube.is_solved()


def evaluate(num_trials=100, scramble_len=20):
    success = 0

    for i in range(num_trials):
        solved = solve_cube(scramble_len)

        if solved:
            success += 1

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_trials} | Success: {success}")

    success_rate = success / num_trials

    print("\n===== RESULTS =====")
    print(f"Trials: {num_trials}")
    print(f"Solved: {success}")
    print(f"Success Rate: {success_rate * 100:.2f}%")

    return success_rate




if __name__ == "__main__":
    evaluate(num_trials=10000, scramble_len=25)
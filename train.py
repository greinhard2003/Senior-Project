import os
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# import your envs
from stage0_env import CubeEnv as Stage0Env
from stage1_env import CubeEnv as Stage1Env
from stage2_env import CubeEnv as Stage2Env
from stage3_env import CubeEnv as Stage3Env


TOTAL_STEPS = 50_000_000
N_ENVS = 8


def make_env(env_class, seed, load_fn=None):
    def _init():
        env = env_class()

        if load_fn is not None:
            try:
                load_fn(env)
            except Exception as e:
                print(f"Warning: failed to load buffer: {e}")

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def train_stage(env_class, model_path, buffer_load_fn=None, buffer_save_fn=None, buffer_attr=None):
    env = SubprocVecEnv([
        make_env(env_class, i, buffer_load_fn) for i in range(N_ENVS)
    ])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    model.save(model_path)


    if buffer_save_fn is not None and buffer_attr is not None:
        buffers = env.get_attr(buffer_attr)

        merged = []
        for b in buffers:
            merged.extend(b)

        print(f"Saving buffer: {buffer_attr}, size={len(merged)}")

        try:
            buffer_save_fn(merged)
        except Exception as e:
            print(f"Warning: failed to save buffer: {e}")

    env.close()


def main():
    os.makedirs("multi_models", exist_ok=True)

    # -------- Stage 0 --------
    print("=== Training Stage 0 ===")

    def save_stage0(buffer):
        with open("stage0_buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

    train_stage(
        Stage0Env,
        "multi_models/stage0",
        buffer_save_fn=save_stage0,
        buffer_attr="stage0_buffer"
    )

    # -------- Stage 1 --------
    print("=== Training Stage 1 ===")

    def load_stage0(env):
        try:
            env.load_stage0_buffer("stage0_buffer.pkl")
        except FileNotFoundError:
            print("Warning: stage0 buffer not found, using random states")

    def save_stage1(buffer):
        with open("stage1_buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

    train_stage(
        Stage1Env,
        "multi_models/stage1",
        buffer_load_fn=load_stage0,
        buffer_save_fn=save_stage1,
        buffer_attr="stage1_buffer"
    )

    # -------- Stage 2 --------
    print("=== Training Stage 2 ===")

    def load_stage1(env):
        try:
            env.load_stage1_buffer("stage1_buffer.pkl")
        except FileNotFoundError:
            print("Warning: stage1 buffer not found, using random states")

    def save_stage2(buffer):
        with open("stage2_buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

    train_stage(
        Stage2Env,
        "multi_models/stage2",
        buffer_load_fn=load_stage1,
        buffer_save_fn=save_stage2,
        buffer_attr="stage2_buffer"
    )

    # -------- Stage 3 --------
    print("=== Training Stage 3 ===")

    def load_stage2(env):
        try:
            env.load_stage2_buffer("stage2_buffer.pkl")
        except FileNotFoundError:
            print("Warning: stage2 buffer not found, using random states")

    train_stage(
        Stage3Env,
        "multi_models/stage3",
        buffer_load_fn=load_stage2
    )


if __name__ == "__main__":
    main()
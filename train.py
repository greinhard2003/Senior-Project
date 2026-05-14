from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import multiprocessing
import torch
from env import CubeEnv
from curriculum import SuccessCurriculumCallback
import sys

torch.set_num_threads(1)

def make_env():
    def _init():
        env = CubeEnv(scramble_len=5, max_steps=60)
        return Monitor(env)
    return _init
 # no Monitor needed

if __name__ == "__main__":
    TARGET_TOTAL_STEPS = 50_000_000
    LOAD_CHECKPOINT = "--resume" in sys.argv
    CHECKPOINT_PATH = "./ppo_cube_CURRENTBEST.zip"
    n_envs = 8  # or multiprocessing.cpu_count()

    train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = CubeEnv(scramble_len=5, max_steps=60)

    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints",
        name_prefix="ppo_cube",
    )

    curriculum = SuccessCurriculumCallback(
        eval_env=eval_env,
        eval_episodes=50,
        eval_freq=100_000,
        solve_threshold=0.70,
        start_scramble=5,
        end_scramble=30,
        scramble_step=2,
        max_steps_scale=8,
        deterministic=True,
        verbose=1,
    )

    if LOAD_CHECKPOINT:
        print("Loading model from checkpoint...")
        model = PPO.load(
            CHECKPOINT_PATH,
            env=train_env,
            device="cpu",
        )
    else:
        print("Starting new model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            device="cpu",
            verbose=0,
            n_steps=256,
            batch_size=512,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
        )

    if LOAD_CHECKPOINT:
        current_steps = model.num_timesteps
        remaining_steps = max(0, TARGET_TOTAL_STEPS - current_steps)

        print(f"Resuming from {current_steps:,} steps")
        print(f"Training for {remaining_steps:,} more steps")

        if remaining_steps == 0:
            print("Target already reached. Exiting.")
        else:
            model.learn(
                total_timesteps=remaining_steps,
                progress_bar=True,
                callback=[checkpoint, curriculum],
                reset_num_timesteps=False,
            )
    else:
        model.learn(
            total_timesteps=TARGET_TOTAL_STEPS,
            progress_bar=True,
            callback=[checkpoint, curriculum],
            reset_num_timesteps=True,
        )

    model.save("ppo_cube_50Mil_CB")

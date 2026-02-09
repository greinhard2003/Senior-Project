from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env import CubeEnv

env = CubeEnv(scramble_len=5, max_steps=50)

# Optional but recommended: validates your env API
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    device="cpu",
    verbose=0,
    n_steps=2048,
    batch_size=256,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
)

checkpoint = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints",
    name_prefix="ppo_cube_5"
)

model.learn(total_timesteps=50_000_000,
            progress_bar=True)
model.save("ppo_cube_5_scramble_50M")

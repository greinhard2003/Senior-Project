from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env import CubeEnv
from curriculum import SuccessCurriculumCallback  # wherever you put it

train_env = Monitor(CubeEnv(scramble_len=5, max_steps=50))
eval_env = CubeEnv(scramble_len=5, max_steps=50)  # no Monitor needed

checkpoint = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints",
    name_prefix="ppo_cube",
)

curriculum = SuccessCurriculumCallback(
    eval_env=eval_env,
    eval_episodes=50,
    eval_freq=200_000,
    solve_threshold=0.30,
    start_scramble=5,
    end_scramble=30,
    scramble_step=2,
    max_steps_scale=8,
    deterministic=True,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    train_env,
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

model.learn(
    total_timesteps=10_000_000,
    progress_bar=True,
    callback=[checkpoint, curriculum],
)

model.save("ppo_cube_curriculumB")

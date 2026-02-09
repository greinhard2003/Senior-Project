from stable_baselines3 import PPO
from env import CubeEnv

model = PPO.load("ppo_cube_5_scramble_50M")
def evaluate(scramble_len, n=100):
    env = CubeEnv(scramble_len=scramble_len, max_steps=150)
    solved = 0
    steps = 0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        solved += int(info["solved"])
        steps += info["steps"]
    print(f"scramble={scramble_len} | solve_rate={solved/n:.2f} | avg_steps={steps/max(solved,1):.1f}")
for s in [1,2,3,5]:
    evaluate(s, n=1000)

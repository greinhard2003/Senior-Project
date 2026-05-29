from stable_baselines3 import PPO
from env import CubeEnv

model = PPO.load("BufferedStagedModel.zip")
def evaluate(scramble_len, n=1000):
    env = CubeEnv(scramble_len=scramble_len, max_steps=500)
    env.target_stage = 4  # require full solve

    solved = 0
    total_steps = 0
    timeouts = 0

    for _ in range(n):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc

        total_steps += info["steps"]

        if info["solved"]:
            solved += 1

        if trunc:
            timeouts += 1

    print(
        f"scramble={scramble_len} | "
        f"solve_rate={solved/n:.2f} | "
        f"timeout_rate={timeouts/n:.2f} | "
        f"avg_steps={total_steps/n:.1f}"
    )
for s in [1,2,3,5, 7, 10, 12, 15, 17, 20, 25]:
    evaluate(s, n=1000)

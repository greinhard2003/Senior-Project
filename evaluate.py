from stable_baselines3 import PPO
from env import CubeEnv

model = PPO.load("ppo_cube_5_scramble_300")
def evaluate(scramble_len, n=100):
    env = CubeEnv(scramble_len=scramble_len, max_steps=500)

    solved = 0
    total_steps = 0
    solved_steps = 0
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
            solved_steps += info["steps"]

        if trunc:
            timeouts += 1

    solve_rate = solved / n
    avg_steps_all = total_steps / n
    avg_steps_solved = solved_steps / solved if solved > 0 else None

    print(
        f"scramble={scramble_len} | "
        f"solve_rate={solve_rate:.2f} | "
        f"avg_all={avg_steps_all:.1f} | "
        f"avg_solved={avg_steps_solved if avg_steps_solved is not None else 'N/A'} | "
        f"timeout_rate={timeouts/n:.2f}"
    )
for s in [1,2,3,5, 7, 10, 12, 15, 17, 20, 25]:
    evaluate(s, n=1000)

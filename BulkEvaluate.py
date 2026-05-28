from stable_baselines3 import PPO
from env import CubeEnv

from pathlib import Path
import csv

MODEL_DIR = "checkpoints"
OUTPUT_CSV = "evaluation_results.csv"

SCRAMBLES = [1, 2, 3, 5, 7, 10, 12, 15, 17, 20, 25]
EPISODES = 100


def evaluate_model(model, scramble_len, n=100):
    env = CubeEnv(scramble_len=scramble_len, max_steps=150)

    solved = 0
    total_steps = 0

    for _ in range(n):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

        solved += int(info["solved"])
        total_steps += info["steps"]

    solve_rate = solved / n
    avg_steps = total_steps / max(solved, 1)

    return solve_rate, avg_steps


# Load all models
model_paths = sorted(Path(MODEL_DIR).glob("*.zip"))

results = []
model_scores = []

for model_path in model_paths:
    print(f"\n=== Evaluating {model_path.name} ===")

    model = PPO.load(str(model_path))

    total_score = 0

    for scramble in SCRAMBLES:
        solve_rate, avg_steps = evaluate_model(
            model,
            scramble_len=scramble,
            n=EPISODES
        )

        print(
            f"scramble={scramble:<2} | "
            f"solve_rate={solve_rate:.3f} | "
            f"avg_steps={avg_steps:.1f}"
        )

        results.append({
            "model": model_path.name,
            "scramble": scramble,
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
        })

        # Simple scoring:
        # prioritize solve rate heavily
        # slight reward for fewer steps
        score = solve_rate - (avg_steps / 1000)
        total_score += score

    avg_score = total_score / len(SCRAMBLES)

    model_scores.append({
        "model": model_path.name,
        "avg_score": avg_score
    })

    print(f"Average score: {avg_score:.4f}")


# Save detailed CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "scramble", "solve_rate", "avg_steps"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to {OUTPUT_CSV}")


# Rank models
top_models = sorted(
    model_scores,
    key=lambda x: x["avg_score"],
    reverse=True
)[:3]

print("\n=== TOP 3 MODELS ===")

for i, entry in enumerate(top_models, start=1):
    print(
        f"{i}. {entry['model']} "
        f"(avg_score={entry['avg_score']:.4f})"
    )
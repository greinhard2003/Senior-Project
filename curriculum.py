import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_episodes=50,
        eval_freq=200_000,
        solve_threshold=0.30,   # raise difficulty when solved >= 30% of eval episodes
        start_scramble=5,
        end_scramble=30,
        scramble_step=2,
        max_steps_scale=8,      # max_steps = max(50, max_steps_scale * scramble_max)
        deterministic=True,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = int(eval_episodes)
        self.eval_freq = int(eval_freq)
        self.solve_threshold = float(solve_threshold)
        self.start_scramble = int(start_scramble)
        self.end_scramble = int(end_scramble)
        self.scramble_step = int(scramble_step)
        self.max_steps_scale = int(max_steps_scale)
        self.deterministic = bool(deterministic)

        self._next_eval = self.eval_freq
        self.current_scramble = self.start_scramble

    def _on_training_start(self) -> None:
        is_resume = self.num_timesteps > 0

        if not is_resume:
            self.current_scramble = self.start_scramble

            if self.verbose:
                print(f"[Curriculum] Fresh start scramble_max={self.current_scramble}")
        else:
            if self.verbose:
                print(f"[Curriculum] Resuming with scramble_max={self.current_scramble}")

        # Apply stored scramble to envs
        self.training_env.set_attr("scramble_max", self.current_scramble)
        self.eval_env.scramble_max = self.current_scramble

        if self.max_steps_scale is not None:
            max_steps = max(50, self.max_steps_scale * self.current_scramble)
            self.training_env.set_attr("max_steps", max_steps)
            self.eval_env.max_steps = max_steps

        # Resume-safe eval scheduling
        if self.num_timesteps > 0:
            self._next_eval = (
                    (self.num_timesteps // self.eval_freq + 1) * self.eval_freq
            )
        else:
            self._next_eval = self.eval_freq

    def _evaluate(self):
        solved = 0

        for ep in range(self.eval_episodes):
            obs, info = self.eval_env.reset(seed=ep)  # deterministic eval
            done = False
            truncated = False

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, truncated, info = self.eval_env.step(action)

            if info.get("solved", False):
                solved += 1

        return solved / self.eval_episodes

    def _maybe_increase_difficulty(self, solve_rate: float):
        current = self.current_scramble

        if solve_rate >= self.solve_threshold and current < self.end_scramble:
            new = min(self.end_scramble, current + self.scramble_step)

            self.current_scramble = new

            self.training_env.set_attr("scramble_max", new)
            self.eval_env.scramble_max = new

            if self.max_steps_scale is not None:
                new_max_steps = max(50, self.max_steps_scale * new)
                self.training_env.set_attr("max_steps", new_max_steps)
                self.eval_env.max_steps = new_max_steps

            if self.verbose:
                print(
                    f"[Curriculum] solve_rate={solve_rate:.2f} >= "
                    f"{self.solve_threshold:.2f} -> scramble_max {current} -> {new}"
                )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            solve_rate = self._evaluate()
            if self.verbose:
                current = self.training_env.get_attr("scramble_max")[0]
                print(f"[Eval] timesteps={self.num_timesteps} scramble_max={current} solve_rate={solve_rate:.2f}")

            self._maybe_increase_difficulty(solve_rate)
            self._next_eval += self.eval_freq

        return True

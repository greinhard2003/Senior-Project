import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SuccessCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_episodes=50,
        eval_freq=200_000,
        solve_threshold=0.70,
        start_scramble=5,
        end_scramble=30,
        scramble_step=2,
        max_steps_scale=8,
        start_stage=0,
        end_stage=3,
        min_evals_before_advance=10,
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

        self.start_stage = int(start_stage)
        self.end_stage = int(end_stage)
        self.min_evals_before_advance = int(min_evals_before_advance)

        self.deterministic = bool(deterministic)

        self._next_eval = self.eval_freq
        self.current_scramble = self.start_scramble
        self.current_stage = self.start_stage

        self.evals_at_current_level = 0

    def _apply_curriculum(self):
        self.training_env.set_attr("scramble_max", self.current_scramble)
        self.training_env.set_attr("target_stage", self.current_stage)

        self.eval_env.scramble_max = self.current_scramble
        self.eval_env.target_stage = self.current_stage

        if self.max_steps_scale is not None:
            max_steps = max(50, self.max_steps_scale * self.current_scramble)
            self.training_env.set_attr("max_steps", max_steps)
            self.eval_env.max_steps = max_steps

    def _on_training_start(self) -> None:
        is_resume = self.num_timesteps > 0

        if not is_resume:
            self.current_scramble = self.start_scramble
            self.current_stage = self.start_stage
            self.evals_at_current_level = 0

            if self.verbose:
                print(
                    f"[Curriculum] Fresh start "
                    f"target_stage={self.current_stage} "
                    f"scramble_max={self.current_scramble}"
                )
        else:
            if self.verbose:
                print(
                    f"[Curriculum] Resuming "
                    f"target_stage={self.current_stage} "
                    f"scramble_max={self.current_scramble}"
                )

        self._apply_curriculum()

        if self.num_timesteps > 0:
            self._next_eval = (
                (self.num_timesteps // self.eval_freq + 1) * self.eval_freq
            )
        else:
            self._next_eval = self.eval_freq

    def _evaluate(self):
        successes = 0

        for ep in range(self.eval_episodes):
            obs, info = self.eval_env.reset(seed=ep)

            done = False
            truncated = False

            while not (done or truncated):
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                )

                obs, reward, done, truncated, info = self.eval_env.step(action)

            reached_target_stage = info.get("stage", 0) > self.current_stage
            solved = info.get("solved", False)

            if reached_target_stage or solved:
                successes += 1

        return successes / self.eval_episodes

    def _maybe_increase_difficulty(self, success_rate: float):
        if self.evals_at_current_level < self.min_evals_before_advance:
            if self.verbose:
                print(
                    f"[Curriculum] Holding level: "
                    f"{self.evals_at_current_level}/"
                    f"{self.min_evals_before_advance} evals completed"
                )
            return

        if success_rate < self.solve_threshold:
            return

        old_stage = self.current_stage
        old_scramble = self.current_scramble

        if self.current_stage < self.end_stage:
            self.current_stage += 1
            changed = True
        elif self.current_scramble < self.end_scramble:
            self.current_scramble = min(
                self.end_scramble,
                self.current_scramble + self.scramble_step,
            )
            changed = True
        else:
            changed = False

        if changed:
            self.evals_at_current_level = 0
            self._apply_curriculum()

            if self.verbose:
                print(
                    f"[Curriculum] success_rate={success_rate:.2f} >= "
                    f"{self.solve_threshold:.2f} -> "
                    f"target_stage {old_stage} -> {self.current_stage}, "
                    f"scramble_max {old_scramble} -> {self.current_scramble}"
                )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            success_rate = self._evaluate()
            self.evals_at_current_level += 1

            if self.verbose:
                print(
                    f"[Eval] timesteps={self.num_timesteps} "
                    f"target_stage={self.current_stage} "
                    f"scramble_max={self.current_scramble} "
                    f"evals_at_level={self.evals_at_current_level} "
                    f"success_rate={success_rate:.2f}"
                )

            self._maybe_increase_difficulty(success_rate)
            self._next_eval += self.eval_freq

        return True
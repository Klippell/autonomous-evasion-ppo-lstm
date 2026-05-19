"""Train the Webots evader with recurrent PPO."""
import argparse
import os
import sys
import time
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

import controllers.evader_env
from controllers.experiment_config import (
    DEFAULT_CONFIG_PATH,
    env_kwargs_from_config,
    load_experiment_config,
    model_kwargs_from_config,
    model_policy_from_config,
)


class MetricCheckpointCallback(BaseCallback):
    """Save checkpoints with rolling episode metrics in the zip name."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str,
        metric_window: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.episode_rewards: deque[float] = deque(maxlen=metric_window)
        self.episode_lengths: deque[int] = deque(maxlen=metric_window)
        self.last_checkpoint_step = 0
        self.best_mean_reward = float("-inf")

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_rewards.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))

        if self.num_timesteps - self.last_checkpoint_step >= self.save_freq:
            self.last_checkpoint_step = self.num_timesteps
            self._save_metric_checkpoint()
        return True

    def _save_metric_checkpoint(self) -> None:
        mean_reward = self._mean_or_zero(self.episode_rewards)
        mean_length = self._mean_or_zero(self.episode_lengths)
        metric_name = (
            f"{self.name_prefix}"
            f"_{self.num_timesteps}_steps"
            f"_rew_{self._metric_token(mean_reward)}"
            f"_len_{mean_length:.0f}"
        )
        checkpoint_path = os.path.join(self.save_path, metric_name)
        self.model.save(checkpoint_path)

        if mean_reward > self.best_mean_reward and len(self.episode_rewards) > 0:
            self.best_mean_reward = mean_reward
            best_name = f"best_{metric_name}"
            self.model.save(os.path.join(self.save_path, best_name))

        if self.verbose > 0:
            print(f"Saved checkpoint: {checkpoint_path}.zip")

    @staticmethod
    def _mean_or_zero(values: deque[float] | deque[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _metric_token(value: float) -> str:
        sign = "p" if value >= 0.0 else "m"
        return f"{sign}{abs(value):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an evader that escapes a pursuer while avoiding obstacles.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a JSON experiment config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training.timesteps from the config.")
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs"))
    parser.add_argument("--save-name", default=None, help="Override training.save_name from the config.")
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    parser.add_argument("--checkpoint-freq", type=int, default=None, help="Override training.checkpoint_freq.")
    parser.add_argument("--metric-window", type=int, default=None, help="Override training.metric_window.")
    parser.add_argument("--hide-reward-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    training_config = config.get("training", {})
    os.makedirs(args.log_dir, exist_ok=True)

    env_kwargs = env_kwargs_from_config(config)
    env_kwargs.update(
        robot_name=args.robot_name,
        show_reward_display=not args.hide_reward_display,
    )
    env: gym.Env = gym.make(
        "Evader-v0",
        **env_kwargs,
    )
    model: BaseAlgorithm = RecurrentPPO(
        model_policy_from_config(config),
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard_logs"),
        **model_kwargs_from_config(config),
    )

    save_name = args.save_name or training_config.get("save_name", "evader_recurrent_ppo")
    checkpoint_callback = MetricCheckpointCallback(
        save_freq=args.checkpoint_freq or int(training_config.get("checkpoint_freq", 50_000)),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix=save_name,
        metric_window=args.metric_window or int(training_config.get("metric_window", 20)),
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps or int(training_config.get("timesteps", 1_000_000)),
        log_interval=int(training_config.get("log_interval", 10)),
        tb_log_name=time.strftime("%Y%m%d-%H%M%S"),
        callback=checkpoint_callback,
    )
    model.save(os.path.join(args.log_dir, save_name))
    env.close()


if __name__ == "__main__":
    main()

"""Train the Webots evader policy."""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import HumanOutputFormat, Logger, TensorBoardOutputFormat
from stable_baselines3.common.utils import get_latest_run_id

import controllers.evader_env
from controllers.evader_env.webots_runtime import WEBOTS_HOME
from controllers.experiment_config import (
    DEFAULT_CONFIG_PATH,
    env_kwargs_from_config,
    load_experiment_config,
    model_algorithm_from_config,
    model_kwargs_from_config,
    model_policy_from_config,
)


DEFAULT_WORLD_PATH = os.path.join(PROJECT_ROOT, "worlds", "my_city_traffic.wbt")
MODEL_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "recurrent_ppo": RecurrentPPO,
}

INFO_TENSORBOARD_KEYS = (
    "distance_to_pursuer",
    "obstacle_distance",
    "front_obstacle_distance",
    "left_obstacle_distance",
    "right_obstacle_distance",
    "back_obstacle_distance",
    "front_corridor_risk",
    "obstacle_risk_escape_scale",
    "obstacle_safety_active",
    "obstacle_safety_action_delta",
    "captured",
    "obstacle_collision",
    "rollover",
    "touch_contact",
    "raw_touch_sensor_contact",
    "touch_collision_plausible",
    "moved_distance",
    "rewarded_moved_distance",
    "speed_kmh",
    "policy_steering",
    "policy_drive",
    "policy_steering_index",
    "policy_drive_index",
    "quantized_steering",
    "quantized_drive",
    "applied_steering",
    "applied_drive",
    "current_steering",
    "steering_direction_streak",
    "obstacle_turn_direction",
    "pursuer_visible",
    "pursuer_reward_total",
    "raw_distance_delta",
    "distance_delta",
    "obstacle_reward_total",
    "movement_reward_total",
    "stability_reward_total",
    "survival_reward_total",
    "front_blocked_speed_penalty",
    "front_blocked_drive_penalty",
    "front_blocked_straight_penalty",
    "front_blocked_brake_reward",
    "obstacle_action_reward",
    "obstacle_action_penalty",
    "obstacle_target_steering",
    "obstacle_target_drive",
    "obstacle_steering_error",
    "obstacle_drive_error",
    "obstacle_turn_commitment_reward",
    "obstacle_turn_switch_penalty",
    "obstacle_turn_progress_reward",
    "obstacle_wrong_heading_penalty",
    "obstacle_insufficient_turn_penalty",
    "obstacle_heading_progress",
    "obstacle_heading_goal",
    "obstacle_safety_intervention_penalty",
    "front_obstacle_penalty",
    "obstacle_clearance_delta_reward",
    "path_prediction_reward",
    "tilt_penalty",
    "overspeed_penalty",
)


class SafeTensorBoardOutputFormat(TensorBoardOutputFormat):
    """TensorBoard writer that recovers if Windows removes the run directory."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        super().__init__(self.folder)

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        os.makedirs(self.folder, exist_ok=True)
        try:
            super().write(key_values, key_excluded, step)
        except FileNotFoundError as exc:
            print(f"TensorBoard log folder was missing; recreating it and retrying once: {exc}")
            self._reopen_writer()
            try:
                super().write(key_values, key_excluded, step)
            except FileNotFoundError as retry_exc:
                print(f"TensorBoard write failed again; skipping this log batch: {retry_exc}")

    def close(self) -> None:
        try:
            super().close()
        except FileNotFoundError:
            self._is_closed = True

    def _reopen_writer(self) -> None:
        try:
            self.writer.close()
        except FileNotFoundError:
            pass
        os.makedirs(self.folder, exist_ok=True)
        super().__init__(self.folder)


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
        self.last_checkpoint_step = self.model.num_timesteps

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


class InfoTensorboardCallback(BaseCallback):
    """Log selected env info metrics so reward hacking is easier to spot."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self.last_log_step = 0
        self.buffers: dict[str, list[float]] = defaultdict(list)

    def _on_training_start(self) -> None:
        self.last_log_step = self.model.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in INFO_TENSORBOARD_KEYS:
                if key not in info:
                    continue
                try:
                    self.buffers[key].append(float(info[key]))
                except (TypeError, ValueError):
                    continue

        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self.last_log_step = self.num_timesteps
            for key, values in self.buffers.items():
                if values:
                    self.logger.record(f"env/{key}", sum(values) / len(values))
            self.buffers.clear()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an evader that escapes a pursuer while avoiding obstacles.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a JSON experiment config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training.timesteps from the config.")
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs"))
    parser.add_argument("--save-name", default=None, help="Override training.save_name from the config.")
    parser.add_argument("--resume-from", default=None, help="Continue training from an existing .zip checkpoint.")
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "recurrent_ppo"),
        default=None,
        help="Override model.algorithm from the config. Useful when resuming older recurrent checkpoints.",
    )
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="When resuming, start TensorBoard/checkpoint timesteps from zero instead of continuing.",
    )
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    parser.add_argument("--checkpoint-freq", type=int, default=None, help="Override training.checkpoint_freq.")
    parser.add_argument("--metric-window", type=int, default=None, help="Override training.metric_window.")
    parser.add_argument("--info-log-freq", type=int, default=None, help="Override training.info_log_freq.")
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event-file writing if Windows/file-sync issues interrupt training.",
    )
    parser.add_argument("--hide-reward-display", action="store_true")
    parser.add_argument(
        "--obstacle-safety",
        dest="obstacle_safety",
        action="store_true",
        default=None,
        help="Enable the obstacle safety teacher even if the config disables it.",
    )
    parser.add_argument(
        "--no-obstacle-safety",
        dest="obstacle_safety",
        action="store_false",
        help="Disable the obstacle safety teacher even if the config enables it.",
    )
    parser.add_argument("--random-obstacles", action="store_true", help="Randomize configured obstacle DEF nodes on each reset.")
    parser.add_argument(
        "--enriched-random-obstacles",
        action="store_true",
        help=(
            "Randomize obstacles and force a two-obstacle curriculum pattern around the evader "
            "on each reset: front/right, front/left, or left/right."
        ),
    )
    parser.add_argument(
        "--nowebots",
        action="store_true",
        help="Start Webots automatically in fast no-rendering batch mode before training.",
    )
    parser.add_argument("--webots-world", default=DEFAULT_WORLD_PATH, help="World file to open when using --nowebots.")
    parser.add_argument("--webots-exe", default=None, help="Path to webots.exe when using --nowebots.")
    return parser.parse_args()


def start_webots_no_rendering(webots_exe: str | None, world_path: str) -> subprocess.Popen:
    executable = webots_exe or _default_webots_executable()
    command = [
        executable,
        "--mode=fast",
        "--no-rendering",
        "--minimize",
        "--batch",
        "--stdout",
        "--stderr",
        os.path.abspath(world_path),
    ]
    print("Starting Webots without rendering:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    time.sleep(5.0)
    return process


def _default_webots_executable() -> str:
    candidates = [
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "webots.exe"),
        os.path.join(WEBOTS_HOME, "webots.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _model_class_from_config(config: dict) -> type[BaseAlgorithm]:
    algorithm = model_algorithm_from_config(config).lower()
    if algorithm not in MODEL_ALGORITHMS:
        choices = ", ".join(sorted(MODEL_ALGORITHMS))
        raise ValueError(f"Unknown model.algorithm '{algorithm}'. Expected one of: {choices}.")
    return MODEL_ALGORITHMS[algorithm]


def _configure_training_logger(
    log_dir: str,
    tb_log_name: str,
    reset_num_timesteps: bool,
    enable_tensorboard: bool,
) -> Logger:
    output_formats = [HumanOutputFormat(sys.stdout)]
    run_dir: str | None = None
    if enable_tensorboard:
        tensorboard_root = os.path.join(log_dir, "tensorboard_logs")
        os.makedirs(tensorboard_root, exist_ok=True)
        latest_run_id = get_latest_run_id(tensorboard_root, tb_log_name)
        if reset_num_timesteps or latest_run_id == 0:
            run_id = latest_run_id + 1
        else:
            run_id = latest_run_id
        run_dir = os.path.join(tensorboard_root, f"{tb_log_name}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        output_formats.append(SafeTensorBoardOutputFormat(run_dir))
        print(f"Logging to {run_dir}")
    else:
        print("TensorBoard logging disabled for this training run.")
    return Logger(folder=run_dir, output_formats=output_formats)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.algorithm is not None:
        config.setdefault("model", {})["algorithm"] = args.algorithm
    training_config = config.get("training", {})
    model_cls = _model_class_from_config(config)
    os.makedirs(args.log_dir, exist_ok=True)
    webots_process: subprocess.Popen | None = None
    env: gym.Env | None = None

    try:
        if args.nowebots:
            webots_process = start_webots_no_rendering(args.webots_exe, args.webots_world)

        env_kwargs = env_kwargs_from_config(config)
        env_kwargs.update(
            robot_name=args.robot_name,
            show_reward_display=not args.hide_reward_display,
        )
        if args.obstacle_safety is not None:
            env_kwargs["obstacle_safety_enabled"] = args.obstacle_safety
        if args.random_obstacles:
            env_kwargs["randomize_obstacles"] = True
        if args.enriched_random_obstacles:
            env_kwargs["randomize_obstacles"] = True
            env_kwargs["enriched_random_obstacles"] = True
        if env_kwargs.get("randomize_obstacles") or env_kwargs.get("enriched_random_obstacles"):
            env_kwargs["center_spawn_when_random_obstacles"] = True
        env = gym.make(
            "Evader-v0",
            **env_kwargs,
        )
        tensorboard_log = os.path.join(args.log_dir, "tensorboard_logs")
        tb_log_name = time.strftime("%Y%m%d-%H%M%S")
        reset_num_timesteps = args.reset_timesteps or args.resume_from is None
        if args.resume_from:
            print(f"Resuming training from: {os.path.abspath(args.resume_from)}")
            model: BaseAlgorithm = model_cls.load(
                args.resume_from,
                env=env,
                tensorboard_log=tensorboard_log,
                verbose=1,
            )
        else:
            model = model_cls(
                model_policy_from_config(config),
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                **model_kwargs_from_config(config),
            )
        model.set_logger(
            _configure_training_logger(
                args.log_dir,
                tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                enable_tensorboard=not args.no_tensorboard,
            )
        )

        save_name = args.save_name or training_config.get("save_name", "evader_ppo")
        checkpoint_callback = MetricCheckpointCallback(
            save_freq=args.checkpoint_freq or int(training_config.get("checkpoint_freq", 50_000)),
            save_path=os.path.join(args.log_dir, "checkpoints"),
            name_prefix=save_name,
            metric_window=args.metric_window or int(training_config.get("metric_window", 20)),
            verbose=1,
        )
        info_callback = InfoTensorboardCallback(
            log_freq=args.info_log_freq or int(training_config.get("info_log_freq", 1000)),
        )
        model.learn(
            total_timesteps=args.timesteps or int(training_config.get("timesteps", 1_000_000)),
            log_interval=int(training_config.get("log_interval", 10)),
            tb_log_name=tb_log_name,
            callback=CallbackList([checkpoint_callback, info_callback]),
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(os.path.join(args.log_dir, save_name))
    finally:
        if env is not None:
            env.close()
        if webots_process is not None and webots_process.poll() is None:
            print("Stopping Webots.")
            webots_process.terminate()


if __name__ == "__main__":
    main()

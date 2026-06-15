"""Run a trained evader policy in Webots."""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import controllers.evader_env
from controllers.experiment_config import DEFAULT_CONFIG_PATH, env_kwargs_from_config, load_experiment_config


MODEL_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "recurrent_ppo": RecurrentPPO,
}


DEBUG_INFO_KEYS = (
    "distance_to_pursuer",
    "obstacle_distance",
    "front_obstacle_distance",
    "left_obstacle_distance",
    "right_obstacle_distance",
    "back_obstacle_distance",
    "front_corridor_risk",
    "obstacle_risk_escape_scale",
    "min_lidar",
    "front_lidar",
    "pursuer_visible",
    "front_pursuer_visible",
    "left_pursuer_visible",
    "right_pursuer_visible",
    "back_pursuer_visible",
    "pursuer_reward_total",
    "obstacle_reward_total",
    "movement_reward_total",
    "stability_reward_total",
    "survival_reward_total",
    "front_obstacle_penalty",
    "side_obstacle_penalty",
    "back_obstacle_penalty",
    "obstacle_clearance_delta_reward",
    "obstacle_collision_free_reward",
    "back_approach_penalty",
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
    "moving_away_reward",
    "visual_moving_away_reward",
    "radial_escape_reward",
    "tangential_orbit_penalty",
    "raw_distance_delta",
    "distance_delta",
    "exploration_reward",
    "movement_reward",
    "rewarded_moved_distance",
    "still_penalty",
    "obstacle_stall_penalty",
    "fast_turn_penalty",
    "tight_turn_penalty",
    "long_curve_penalty",
    "straight_drive_reward",
    "clear_path_straight_reward",
    "unnecessary_turn_penalty",
    "avoidance_turn_reward",
    "over_avoidance_turn_penalty",
    "clear_front_turn_penalty",
    "straighten_reward",
    "turn_towards_obstacle_penalty",
    "turn_towards_visible_pursuer_penalty",
    "path_prediction_reward",
    "action_smoothness_penalty",
    "tilt_penalty",
    "overspeed_penalty",
    "survival_reward",
    "captured",
    "obstacle_collision",
    "rollover",
    "touch_contact",
    "raw_touch_sensor_contact",
    "touch_collision_plausible",
    "moved_distance",
    "speed_kmh",
    "target_speed",
    "throttle",
    "brake",
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
    "obstacle_safety_active",
    "obstacle_safety_action_delta",
    "evader_steering_target",
    "evader_steering_command_target",
    "evader_steering_stabilization_active",
    "evader_steering_stabilization_delta",
    "evader_steering_commit_direction",
    "evader_steering_commit_steps_left",
    "pursuer_avoidance_active",
    "pursuer_avoidance_obstacle_count",
    "pursuer_avoidance_side",
    "pursuer_avoidance_commit_steps_left",
    "pursuer_return_to_chase_steps_left",
    "pursuer_lidar_avoidance_active",
    "pursuer_lidar_front_distance",
    "pursuer_lidar_left_distance",
    "pursuer_lidar_right_distance",
    "pursuer_lidar_danger_count",
    "pursuer_planner_active",
    "pursuer_planner_path_length",
    "pursuer_planner_replan",
    "pursuer_planner_stuck_recovery",
    "pursuer_count",
    "pursuer_direct_chase_hold_steps_left",
    "pursuer_behavior_limited",
    "pursuer_line_of_sight",
    "pursuer_info_mode",
    "pursuer_hint_refresh",
    "pursuer_hint_age_seconds",
    "timing_total_ms",
    "timing_control_ms",
    "timing_pursuer_ms",
    "timing_webots_ms",
    "timing_webots_max_step_ms",
    "timing_webots_steps",
    "timing_observation_ms",
    "timing_reward_ms",
    "timing_finish_ms",
    "timing_display_ms",
)

OBS_FIELD_NAMES = (
    *(f"obs_lidar_{index:02d}" for index in range(12)),
    "obs_lidar_min",
    "obs_front_lidar_min",
    "obs_left_lidar_min",
    "obs_back_lidar_min",
    "obs_right_lidar_min",
    "obs_vision_visible",
    "obs_pursuer_dx",
    "obs_pursuer_dy",
    "obs_pursuer_distance",
    "obs_pursuer_sin_bearing",
    "obs_ego_speed",
    "obs_ego_acceleration",
    "obs_ego_steering",
    "obs_ego_yaw_rate",
    "obs_ego_heading_sin",
    "obs_ego_heading_cos",
    "obs_ego_touch",
    "obs_avoidance_front_risk",
    "obs_avoidance_danger_delta",
    "obs_avoidance_turn_direction",
    "obs_avoidance_heading_progress",
    "obs_avoidance_heading_remaining",
)

DEBUG_FIELD_NAMES = (
    "episode",
    "episode_step",
    "global_step",
    "reward",
    "terminated",
    "truncated",
    "model_steering",
    "model_drive",
    *OBS_FIELD_NAMES,
    *DEBUG_INFO_KEYS,
)


def _space_summary(space: gym.Space) -> str:
    if isinstance(space, gym.spaces.Dict):
        return "Dict(" + ", ".join(f"{key}:{value.shape}" for key, value in space.spaces.items()) + ")"
    return f"{type(space).__name__}{getattr(space, 'shape', '')}"


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _action_values(action: np.ndarray) -> tuple[float, float]:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    steering = float(action_array[0]) if action_array.size > 0 else 0.0
    drive = float(action_array[1]) if action_array.size > 1 else 0.0
    return steering, drive


def _obs_debug_values(obs: dict[str, np.ndarray]) -> dict[str, float]:
    values: dict[str, float] = {field_name: 0.0 for field_name in OBS_FIELD_NAMES}

    lidar = np.asarray(obs.get("lidar", np.ones(12, dtype=np.float32)), dtype=np.float32).reshape(-1)
    for index in range(min(12, lidar.size)):
        values[f"obs_lidar_{index:02d}"] = float(lidar[index])
    if lidar.size:
        values["obs_lidar_min"] = float(np.min(lidar))
    if lidar.size >= 12:
        values["obs_front_lidar_min"] = float(np.min(lidar[0:3]))
        values["obs_left_lidar_min"] = float(np.min(lidar[3:6]))
        values["obs_back_lidar_min"] = float(np.min(lidar[6:9]))
        values["obs_right_lidar_min"] = float(np.min(lidar[9:12]))

    vision = np.asarray(obs.get("vision", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if vision.size:
        values["obs_vision_visible"] = float(vision[0])

    pursuer = np.asarray(obs.get("pursuer", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)
    for key, value in zip(
        ("obs_pursuer_dx", "obs_pursuer_dy", "obs_pursuer_distance", "obs_pursuer_sin_bearing"),
        pursuer,
    ):
        values[key] = float(value)

    ego = np.asarray(obs.get("ego", np.zeros(7, dtype=np.float32)), dtype=np.float32).reshape(-1)
    for key, value in zip(
        (
            "obs_ego_speed",
            "obs_ego_acceleration",
            "obs_ego_steering",
            "obs_ego_yaw_rate",
            "obs_ego_heading_sin",
            "obs_ego_heading_cos",
            "obs_ego_touch",
        ),
        ego,
    ):
        values[key] = float(value)

    avoidance = np.asarray(obs.get("avoidance", np.zeros(5, dtype=np.float32)), dtype=np.float32).reshape(-1)
    for key, value in zip(
        (
            "obs_avoidance_front_risk",
            "obs_avoidance_danger_delta",
            "obs_avoidance_turn_direction",
            "obs_avoidance_heading_progress",
            "obs_avoidance_heading_remaining",
        ),
        avoidance,
    ):
        values[key] = float(value)
    return values


class DebugReport:
    """Write a step-by-step inference report and print suspicious moments."""

    def __init__(self, path: str, print_interval: int = 10, rolling_window: int = 100) -> None:
        self.path = path
        self.print_interval = max(1, int(print_interval))
        self.rolling_window = max(1, int(rolling_window))
        self.global_step = 0
        self.episode = 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_values: dict[str, list[float]] = defaultdict(list)
        self.rolling_values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.rolling_window))

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=DEBUG_FIELD_NAMES, extrasaction="ignore")
        self.writer.writeheader()
        print(f"Debug report enabled. Writing every step to: {path}")

    def record(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> None:
        self.global_step += 1
        self.episode_step += 1
        self.episode_reward += float(reward)

        model_steering, model_drive = _action_values(action)
        row: dict[str, float | int] = {
            "episode": self.episode,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "reward": float(reward),
            "terminated": int(terminated),
            "truncated": int(truncated),
            "model_steering": model_steering,
            "model_drive": model_drive,
            **_obs_debug_values(obs),
        }
        for key in DEBUG_INFO_KEYS:
            row[key] = _float_or_zero(info.get(key, 0.0))

        self.writer.writerow(row)
        self._track(row)

        should_print = (
            self.episode_step % self.print_interval == 0
            or terminated
            or truncated
            or bool(row["captured"])
            or bool(row["touch_contact"])
            or row["front_corridor_risk"] > 0.55
            or row["obstacle_distance"] < 2.5
        )
        if should_print:
            self._print_step(row)
            self.file.flush()

        if terminated or truncated:
            self._print_episode_summary(terminated=terminated, truncated=truncated)
            self.episode += 1
            self.episode_step = 0
            self.episode_reward = 0.0
            self.episode_values.clear()

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()
            print(f"Debug report saved: {self.path}")

    def _track(self, row: dict[str, float | int]) -> None:
        for key in (
            "reward",
            "distance_to_pursuer",
            "obstacle_distance",
            "front_corridor_risk",
            "touch_contact",
            "captured",
            "speed_kmh",
            "applied_steering",
            "applied_drive",
            "evader_steering_stabilization_active",
            "evader_steering_stabilization_delta",
            "pursuer_reward_total",
            "obstacle_reward_total",
            "movement_reward_total",
            "stability_reward_total",
            "survival_reward_total",
        ):
            value = float(row.get(key, 0.0))
            self.episode_values[key].append(value)
            self.rolling_values[key].append(value)

    def _print_step(self, row: dict[str, float | int]) -> None:
        rolling_reward = self._rolling_mean("reward")
        print(
            "DBG "
            f"ep {int(row['episode'])} step {int(row['episode_step'])} "
            f"r {float(row['reward']):+.2f} rollR {rolling_reward:+.2f} "
            f"dist {float(row['distance_to_pursuer']):.1f} "
            f"obs F/L/R/B {float(row['front_obstacle_distance']):.1f}/"
            f"{float(row['left_obstacle_distance']):.1f}/"
            f"{float(row['right_obstacle_distance']):.1f}/"
            f"{float(row['back_obstacle_distance']):.1f} "
            f"risk {float(row['front_corridor_risk']):.2f} "
            f"turn {float(row['obstacle_heading_progress']):+.2f}/{float(row['obstacle_heading_goal']):.2f} "
            f"act {float(row['policy_steering']):+.2f}/{float(row['policy_drive']):+.2f} "
            f"idx {int(float(row['policy_steering_index']))}/{int(float(row['policy_drive_index']))} "
            f"q {float(row['quantized_steering']):+.2f} "
            f"app {float(row['applied_steering']):+.2f}/{float(row['applied_drive']):+.2f} "
            f"eSt {int(float(row['evader_steering_stabilization_active']))}/"
            f"{float(row['evader_steering_stabilization_delta']):.2f}/"
            f"{int(float(row['evader_steering_commit_direction']))}/"
            f"{int(float(row['evader_steering_commit_steps_left']))} "
            f"spd {float(row['speed_kmh']):.1f} "
            f"safe {int(float(row['obstacle_safety_active']))}/{float(row['obstacle_safety_action_delta']):.2f} "
            f"pAvoid {int(float(row['pursuer_avoidance_active']))}/{int(float(row['pursuer_avoidance_obstacle_count']))} "
            f"pSide {int(float(row['pursuer_avoidance_side']))}/{int(float(row['pursuer_avoidance_commit_steps_left']))} "
            f"pLid {int(float(row['pursuer_lidar_avoidance_active']))}/"
            f"{float(row['pursuer_lidar_front_distance']):.1f}/"
            f"{float(row['pursuer_lidar_left_distance']):.1f}/"
            f"{float(row['pursuer_lidar_right_distance']):.1f} "
            f"pPlan {int(float(row['pursuer_planner_active']))}/"
            f"{int(float(row['pursuer_planner_path_length']))}/"
            f"{int(float(row['pursuer_planner_replan']))}/"
            f"{int(float(row['pursuer_planner_stuck_recovery']))} "
            f"pInfo {int(float(row['pursuer_info_mode']))}/{int(float(row['pursuer_line_of_sight']))}/{float(row['pursuer_hint_age_seconds']):.0f}s "
            f"tim {float(row['timing_total_ms']):.0f}/"
            f"{float(row['timing_webots_ms']):.0f}/"
            f"{float(row['timing_webots_max_step_ms']):.0f}/"
            f"{float(row['timing_display_ms']):.0f}ms "
            f"touch {int(float(row['touch_contact']))}/{int(float(row['raw_touch_sensor_contact']))}/{int(float(row['touch_collision_plausible']))} "
            f"cap {int(float(row['captured']))} "
            f"grp P/O/M/S/U {float(row['pursuer_reward_total']):+.2f}/"
            f"{float(row['obstacle_reward_total']):+.2f}/"
            f"{float(row['movement_reward_total']):+.2f}/"
            f"{float(row['stability_reward_total']):+.2f}/"
            f"{float(row['survival_reward_total']):+.2f}"
        )

    def _print_episode_summary(self, terminated: bool, truncated: bool) -> None:
        reason = "timeout" if truncated else "terminated"
        if self._episode_mean("captured") > 0.0:
            reason = "captured"
        elif self._episode_mean("touch_contact") > 0.0:
            reason = "touch/collision"

        print(
            "DBG EPISODE "
            f"{self.episode} ended ({reason}; terminated={int(terminated)} truncated={int(truncated)}) "
            f"steps {self.episode_step} totalR {self.episode_reward:+.2f} "
            f"meanDist {self._episode_mean('distance_to_pursuer'):.1f} "
            f"minObs {self._episode_min('obstacle_distance'):.2f} "
            f"meanRisk {self._episode_mean('front_corridor_risk'):.2f} "
            f"meanSpeed {self._episode_mean('speed_kmh'):.1f} "
            f"meanSteer {self._episode_abs_mean('applied_steering'):.2f} "
            f"groups P/O/M/S/U {self._episode_mean('pursuer_reward_total'):+.2f}/"
            f"{self._episode_mean('obstacle_reward_total'):+.2f}/"
            f"{self._episode_mean('movement_reward_total'):+.2f}/"
            f"{self._episode_mean('stability_reward_total'):+.2f}/"
            f"{self._episode_mean('survival_reward_total'):+.2f}"
        )

    def _rolling_mean(self, key: str) -> float:
        values = self.rolling_values[key]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _episode_mean(self, key: str) -> float:
        values = self.episode_values[key]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _episode_abs_mean(self, key: str) -> float:
        values = self.episode_values[key]
        if not values:
            return 0.0
        return float(sum(abs(value) for value in values) / len(values))

    def _episode_min(self, key: str) -> float:
        values = self.episode_values[key]
        if not values:
            return 0.0
        return float(min(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained evader policy.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--model",
        default=os.path.join(PROJECT_ROOT, "logs", "evader_ppo.zip"),
    )
    parser.add_argument("--robot-name", default=os.environ.get("WEBOTS_ROBOT_NAME", "evader"))
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy, matching training-time behavior more closely.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("auto", "ppo", "recurrent_ppo"),
        default="auto",
        help="Algorithm used by the checkpoint. Auto tries PPO first, then recurrent PPO.",
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
    parser.add_argument(
        "--limited-info-pursuer",
        action="store_true",
        help=(
            "Use the experiment pursuer: random spawn, random patrol, noisy direction hints every "
            "10 seconds, and exact pursuit only with line of sight."
        ),
    )
    parser.add_argument(
        "--center-spawn",
        action="store_true",
        help="Use the centered spawn used by random-obstacle training, even without randomizing obstacles.",
    )
    parser.add_argument("--random-obstacles", action="store_true", help="Randomize configured obstacle DEF nodes on each reset.")
    parser.add_argument(
        "--enriched-random-obstacles",
        action="store_true",
        help=(
            "Use the same enriched random obstacle curriculum as training: randomize the map and "
            "force front/right, front/left, or left/right blockers near the evader."
        ),
    )
    parser.add_argument(
        "--debug-report",
        action="store_true",
        help="Print detailed inference diagnostics and save a per-step CSV report.",
    )
    parser.add_argument(
        "--debug-report-interval",
        type=int,
        default=10,
        help="Print one debug line every N inference steps, plus dangerous steps and episode endings.",
    )
    parser.add_argument(
        "--debug-report-path",
        default=None,
        help="CSV path for --debug-report. Defaults to logs/debug_reports/inference_<time>.csv.",
    )
    parser.add_argument(
        "--debug-report-window",
        type=int,
        default=100,
        help="Rolling step window used for debug-report terminal averages.",
    )
    return parser.parse_args()


def _load_model(path: str, algorithm: str) -> tuple[BaseAlgorithm, str]:
    if algorithm != "auto":
        return MODEL_ALGORITHMS[algorithm].load(path), algorithm

    errors: list[str] = []
    for name, model_cls in MODEL_ALGORITHMS.items():
        try:
            return model_cls.load(path), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Could not load checkpoint with any known algorithm:\n{joined_errors}")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    print(f"Loading model: {os.path.abspath(args.model)}")
    print(f"Inference mode: {'stochastic' if args.stochastic else 'deterministic'}")
    model, loaded_algorithm = _load_model(args.model, args.algorithm)
    print(f"Loaded algorithm: {loaded_algorithm}")
    env_kwargs = env_kwargs_from_config(config)
    env_kwargs.update(
        robot_name=args.robot_name,
        show_reward_display=not args.hide_reward_display,
    )
    if args.obstacle_safety is not None:
        env_kwargs["obstacle_safety_enabled"] = args.obstacle_safety
    if args.limited_info_pursuer:
        env_kwargs["pursuer_behavior_mode"] = "limited_info_patrol"
        env_kwargs["pursuer_random_spawn"] = True
    if args.random_obstacles:
        env_kwargs["randomize_obstacles"] = True
    if args.enriched_random_obstacles:
        env_kwargs["randomize_obstacles"] = True
        env_kwargs["enriched_random_obstacles"] = True
    if args.center_spawn:
        env_kwargs["force_center_spawn"] = True
    if env_kwargs.get("randomize_obstacles") or env_kwargs.get("enriched_random_obstacles") or env_kwargs.get("force_center_spawn"):
        env_kwargs["center_spawn_when_random_obstacles"] = True
        print("Using random-obstacle training spawn.")
    env: gym.Env = gym.make(
        "Evader-v0",
        **env_kwargs,
    )
    if model.observation_space != env.observation_space:
        raise RuntimeError(
            "This checkpoint was trained with a different observation space than the current environment.\n"
            f"Model expects: {_space_summary(model.observation_space)}\n"
            f"Current env returns: {_space_summary(env.observation_space)}\n"
            "Use a checkpoint trained after the latest observation changes, or retrain with the current code/config."
        )

    debug_report: DebugReport | None = None
    if args.debug_report:
        report_path = args.debug_report_path
        if report_path is None:
            filename = f"inference_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            report_path = os.path.join(PROJECT_ROOT, "logs", "debug_reports", filename)
        debug_report = DebugReport(
            report_path,
            print_interval=args.debug_report_interval,
            rolling_window=args.debug_report_window,
        )

    obs, _info = env.reset()
    is_recurrent = isinstance(model, RecurrentPPO)
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    try:
        while True:
            current_obs = obs
            if is_recurrent:
                action, lstm_states = model.predict(
                    current_obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=not args.stochastic,
                )
            else:
                action, _states = model.predict(
                    current_obs,
                    deterministic=not args.stochastic,
                )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if debug_report is not None:
                debug_report.record(current_obs, action, reward, terminated, truncated, info)
            if done:
                lstm_states = None
                obs, _info = env.reset()
            if is_recurrent:
                episode_starts = np.array([done], dtype=bool)
    except KeyboardInterrupt:
        print("Inference interrupted.")
    finally:
        if debug_report is not None:
            debug_report.close()
        env.close()


if __name__ == "__main__":
    main()

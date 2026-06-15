import math
import os
import time
import heapq
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from controllers.evader_env.debug_display import DebugDisplayMixin
from controllers.evader_env.reward import RewardMixin, reward_weights_from_mapping
from controllers.evader_env.webots_runtime import DEFAULT_SPAWN_POSES, SpawnPose, configure_webots_paths


@dataclass(frozen=True)
class ObstacleFootprint:
    radius: float
    vertices: tuple[tuple[float, float], ...] = ()
    parts: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass
class WorldFootprint:
    center: np.ndarray
    radius: float
    label: str
    vertices: np.ndarray | None = None
    parts: tuple[np.ndarray, ...] = ()


class EvaderEnv(RewardMixin, DebugDisplayMixin, gym.Env):
    """Webots Gym environment for training an evader against a simple pursuer."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 2500,
        capture_distance: float = 6.0,
        vehicle_touch_distance: float = 5.5,
        touch_collision_lidar_distance: float = 2.0,
        obstacle_collision_distance: float = 1.25,
        capture_termination_penalty: float = 150.0,
        obstacle_collision_termination_penalty: float = 350.0,
        rollover_termination_penalty: float = 250.0,
        rollover_termination_angle: float = 0.85,
        reset_touch_grace_steps: int = 3,
        reset_sensor_warmup_steps: int = 12,
        pursuer_speed_mps: float = 6.0,
        pursuer_count: int = 1,
        pursuer_spawn_ring_radius: float = 12.0,
        evader_speed_margin_mps: float = 2.0,
        robot_name: str | None = None,
        still_distance_threshold: float = 0.1,
        self_lidar_ignore_distance: float = 0.08,
        action_repeat: int = 4,
        discrete_actions: bool = True,
        pursuer_start_delay_steps: int = 240,
        obstacle_safety_enabled: bool = True,
        obstacle_safety_slow_distance: float = 9.0,
        obstacle_safety_brake_distance: float = 4.5,
        obstacle_safety_min_steering: float = 0.25,
        obstacle_safety_corridor_width: float = 0.45,
        evader_steering_rate_limit: float = 0.12,
        evader_obstacle_steering_commit_steps: int = 8,
        evader_obstacle_steering_commit_risk: float = 0.12,
        evader_obstacle_steering_release_risk: float = 0.06,
        evader_obstacle_steering_min_abs: float = 0.25,
        pursuer_avoid_obstacles: bool = True,
        pursuer_obstacle_margin: float = 2.0,
        pursuer_avoidance_lookahead_steps: float = 5.0,
        pursuer_lidar_avoidance_enabled: bool = True,
        pursuer_lidar_range: float = 28.0,
        pursuer_lidar_danger_distance: float = 14.0,
        pursuer_lidar_ray_count: int = 15,
        pursuer_lidar_scan_angle_degrees: float = 180.0,
        pursuer_lidar_front_angle_degrees: float = 45.0,
        pursuer_avoidance_commit_steps: int = 24,
        pursuer_return_to_chase_steps: int = 18,
        pursuer_avoidance_same_obstacle_distance: float = 10.0,
        pursuer_behavior_mode: str = "direct_chase",
        pursuer_random_spawn: bool = False,
        pursuer_random_spawn_bounds: tuple[float, float, float, float] | list[float] | None = None,
        pursuer_random_spawn_min_evader_distance: float = 35.0,
        pursuer_random_spawn_obstacle_margin: float = 8.0,
        pursuer_spawn_wall_margin: float = 6.0,
        pursuer_limited_info_update_seconds: float = 10.0,
        pursuer_limited_info_direction_noise_degrees: float = 20.0,
        pursuer_limited_info_patrol_update_seconds: float = 2.5,
        pursuer_limited_info_patrol_turn_degrees: float = 35.0,
        pursuer_line_of_sight_max_distance: float = 140.0,
        pursuer_line_of_sight_obstacle_margin: float = 2.0,
        pursuer_planner_enabled: bool = True,
        pursuer_planner_cell_size: float = 4.0,
        pursuer_planner_padding: float = 2.0,
        pursuer_planner_replan_steps: int = 8,
        pursuer_planner_goal_tolerance: float = 8.0,
        pursuer_planner_waypoint_tolerance: float = 3.0,
        pursuer_planner_max_expansions: int = 12000,
        pursuer_direct_chase_hold_steps: int = 12,
        pursuer_unstuck_enabled: bool = True,
        pursuer_unstuck_window_steps: int = 10,
        pursuer_unstuck_min_progress: float = 0.35,
        show_reward_display: bool = True,
        reward_display_interval: int = 1,
        show_car_display: bool = False,
        timing_profile_enabled: bool = True,
        timing_slow_step_seconds: float = 0.75,
        timing_log_interval_steps: int = 25,
        reset_vehicle_physics: bool = True,
        exploration_cell_size: float = 8.0,
        sensor_timestep: int | None = None,
        enable_camera_recognition: bool = True,
        randomize_obstacles: bool = False,
        randomize_all_buildings: bool = True,
        obstacle_excluded_type_names: tuple[str, ...] | list[str] = ("TheThreeTowers", "Auditorium"),
        obstacle_excluded_def_names: tuple[str, ...] | list[str] = (),
        enriched_random_obstacles: bool = False,
        enriched_random_patterns: tuple[str, ...] | list[str] = ("front_right", "front_left", "left_right"),
        enriched_random_obstacle_def_names: tuple[str, ...] | list[str] = (),
        enriched_random_front_distance: float = 12.0,
        enriched_random_side_distance: float = 8.0,
        enriched_random_jitter: float = 1.5,
        enriched_random_evader_clearance: float = 5.0,
        center_spawn_when_random_obstacles: bool = True,
        force_center_spawn: bool = False,
        random_obstacle_def_names: tuple[str, ...] | list[str] = (),
        random_obstacle_bounds: tuple[float, float, float, float] | list[float] = (-170.0, 170.0, -190.0, 260.0),
        random_obstacle_exclusion_center: tuple[float, float] | list[float] = (0.0, 0.0),
        random_obstacle_exclusion_radius: float = 30.0,
        random_obstacle_min_spacing: float = 12.0,
        random_obstacle_edge_spacing: float = 3.0,
        reward_weights: dict[str, object] | None = None,
        front_camera_names: tuple[str, ...] | list[str] = ("front camera", "front Camera", "camera"),
        back_camera_names: tuple[str, ...] | list[str] = ("back camera", "rear camera"),
        left_camera_names: tuple[str, ...] | list[str] = ("left camera",),
        right_camera_names: tuple[str, ...] | list[str] = ("right camera",),
        pursuer_recognition_tokens: tuple[str, ...] | list[str] = ("pursuer", "Pursuer"),
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.capture_distance = capture_distance
        self.vehicle_touch_distance = vehicle_touch_distance
        self.touch_collision_lidar_distance = max(float(touch_collision_lidar_distance), 0.0)
        self.obstacle_collision_distance = max(float(obstacle_collision_distance), 0.0)
        self.capture_termination_penalty = max(float(capture_termination_penalty), 0.0)
        self.obstacle_collision_termination_penalty = max(float(obstacle_collision_termination_penalty), 0.0)
        self.rollover_termination_penalty = max(float(rollover_termination_penalty), 0.0)
        self.rollover_termination_angle = max(float(rollover_termination_angle), 0.0)
        self.reset_touch_grace_steps = max(int(reset_touch_grace_steps), 0)
        self.reset_sensor_warmup_steps = max(int(reset_sensor_warmup_steps), 0)
        self.pursuer_speed_mps = pursuer_speed_mps
        self.pursuer_count = max(int(pursuer_count), 1)
        self.pursuer_spawn_ring_radius = max(float(pursuer_spawn_ring_radius), 0.0)
        self.evader_speed_margin_mps = evader_speed_margin_mps
        self.robot_name = robot_name or os.environ.get("WEBOTS_ROBOT_NAME") or "evader"
        self.still_distance_threshold = still_distance_threshold
        self.self_lidar_ignore_distance = self_lidar_ignore_distance
        self.action_repeat = max(int(action_repeat), 1)
        self.discrete_actions = bool(discrete_actions)
        self.pursuer_start_delay_steps = pursuer_start_delay_steps
        self.obstacle_safety_enabled = obstacle_safety_enabled
        self.obstacle_safety_slow_distance = float(obstacle_safety_slow_distance)
        self.obstacle_safety_brake_distance = float(obstacle_safety_brake_distance)
        self.obstacle_safety_min_steering = float(obstacle_safety_min_steering)
        self.obstacle_safety_corridor_width = float(obstacle_safety_corridor_width)
        self.evader_steering_rate_limit = max(float(evader_steering_rate_limit), 0.0)
        self.evader_obstacle_steering_commit_steps = max(int(evader_obstacle_steering_commit_steps), 0)
        self.evader_obstacle_steering_commit_risk = float(np.clip(evader_obstacle_steering_commit_risk, 0.0, 1.0))
        self.evader_obstacle_steering_release_risk = float(np.clip(evader_obstacle_steering_release_risk, 0.0, 1.0))
        self.evader_obstacle_steering_min_abs = float(np.clip(evader_obstacle_steering_min_abs, 0.0, 0.5))
        self.pursuer_avoid_obstacles = bool(pursuer_avoid_obstacles)
        self.pursuer_obstacle_margin = max(float(pursuer_obstacle_margin), 0.0)
        self.pursuer_avoidance_lookahead_steps = max(float(pursuer_avoidance_lookahead_steps), 1.0)
        self.pursuer_lidar_avoidance_enabled = bool(pursuer_lidar_avoidance_enabled)
        self.pursuer_lidar_range = max(float(pursuer_lidar_range), 1.0)
        self.pursuer_lidar_danger_distance = max(float(pursuer_lidar_danger_distance), 1.0)
        self.pursuer_lidar_ray_count = max(int(pursuer_lidar_ray_count), 7)
        self.pursuer_lidar_scan_angle = math.radians(max(float(pursuer_lidar_scan_angle_degrees), 1.0))
        self.pursuer_lidar_front_angle = math.radians(max(float(pursuer_lidar_front_angle_degrees), 1.0))
        self.pursuer_avoidance_commit_steps = max(int(pursuer_avoidance_commit_steps), 0)
        self.pursuer_return_to_chase_steps = max(int(pursuer_return_to_chase_steps), 0)
        self.pursuer_avoidance_same_obstacle_distance = max(float(pursuer_avoidance_same_obstacle_distance), 0.0)
        self.pursuer_behavior_mode = str(pursuer_behavior_mode).lower()
        if self.pursuer_behavior_mode not in {"direct_chase", "limited_info_patrol"}:
            raise ValueError("pursuer_behavior_mode must be 'direct_chase' or 'limited_info_patrol'.")
        self.pursuer_random_spawn = bool(pursuer_random_spawn or self.pursuer_behavior_mode == "limited_info_patrol")
        spawn_bounds = pursuer_random_spawn_bounds if pursuer_random_spawn_bounds is not None else random_obstacle_bounds
        self.pursuer_random_spawn_bounds = tuple(float(value) for value in spawn_bounds)
        self.pursuer_random_spawn_min_evader_distance = max(float(pursuer_random_spawn_min_evader_distance), 0.0)
        self.pursuer_random_spawn_obstacle_margin = max(float(pursuer_random_spawn_obstacle_margin), 0.0)
        self.pursuer_spawn_wall_margin = max(float(pursuer_spawn_wall_margin), 0.0)
        self.pursuer_limited_info_update_seconds = max(float(pursuer_limited_info_update_seconds), 0.1)
        self.pursuer_limited_info_direction_noise_degrees = max(float(pursuer_limited_info_direction_noise_degrees), 0.0)
        self.pursuer_limited_info_patrol_update_seconds = max(float(pursuer_limited_info_patrol_update_seconds), 0.1)
        self.pursuer_limited_info_patrol_turn_degrees = max(float(pursuer_limited_info_patrol_turn_degrees), 0.0)
        self.pursuer_line_of_sight_max_distance = max(float(pursuer_line_of_sight_max_distance), 0.0)
        self.pursuer_line_of_sight_obstacle_margin = max(float(pursuer_line_of_sight_obstacle_margin), 0.0)
        self.pursuer_planner_enabled = bool(pursuer_planner_enabled)
        self.pursuer_planner_cell_size = max(float(pursuer_planner_cell_size), 1.0)
        self.pursuer_planner_padding = max(float(pursuer_planner_padding), 0.0)
        self.pursuer_planner_replan_steps = max(int(pursuer_planner_replan_steps), 1)
        self.pursuer_planner_goal_tolerance = max(float(pursuer_planner_goal_tolerance), self.pursuer_planner_cell_size)
        self.pursuer_planner_waypoint_tolerance = max(float(pursuer_planner_waypoint_tolerance), 0.5)
        self.pursuer_planner_max_expansions = max(int(pursuer_planner_max_expansions), 100)
        self.pursuer_direct_chase_hold_steps = max(int(pursuer_direct_chase_hold_steps), 0)
        self.pursuer_unstuck_enabled = bool(pursuer_unstuck_enabled)
        self.pursuer_unstuck_window_steps = max(int(pursuer_unstuck_window_steps), 2)
        self.pursuer_unstuck_min_progress = max(float(pursuer_unstuck_min_progress), 0.0)
        self.show_reward_display = show_reward_display
        self.reward_display_interval = reward_display_interval
        self.show_car_display = show_car_display
        self.timing_profile_enabled = bool(timing_profile_enabled)
        self.timing_slow_step_seconds = max(float(timing_slow_step_seconds), 0.0)
        self.timing_log_interval_steps = max(int(timing_log_interval_steps), 1)
        self.reset_vehicle_physics = reset_vehicle_physics
        self.exploration_cell_size = max(float(exploration_cell_size), 1.0)
        self.sensor_timestep = sensor_timestep
        self.enable_camera_recognition = enable_camera_recognition
        self.randomize_obstacles = randomize_obstacles
        self.randomize_all_buildings = randomize_all_buildings
        self.obstacle_excluded_type_names = {str(name) for name in obstacle_excluded_type_names}
        self.obstacle_excluded_def_names = {str(name) for name in obstacle_excluded_def_names}
        self.enriched_random_obstacles = enriched_random_obstacles
        self.enriched_random_patterns = tuple(enriched_random_patterns)
        self.enriched_random_obstacle_def_names = tuple(enriched_random_obstacle_def_names)
        self.enriched_random_front_distance = float(enriched_random_front_distance)
        self.enriched_random_side_distance = float(enriched_random_side_distance)
        self.enriched_random_jitter = float(enriched_random_jitter)
        self.enriched_random_evader_clearance = float(enriched_random_evader_clearance)
        self.center_spawn_when_random_obstacles = center_spawn_when_random_obstacles
        self.force_center_spawn = force_center_spawn
        self.random_obstacle_def_names = tuple(random_obstacle_def_names)
        self.random_obstacle_bounds = tuple(float(value) for value in random_obstacle_bounds)
        self.random_obstacle_exclusion_center = np.array(random_obstacle_exclusion_center, dtype=np.float32)
        self.random_obstacle_exclusion_radius = float(random_obstacle_exclusion_radius)
        self.random_obstacle_min_spacing = float(random_obstacle_min_spacing)
        self.random_obstacle_edge_spacing = max(float(random_obstacle_edge_spacing), 0.0)
        self.reward_weights = reward_weights_from_mapping(reward_weights)
        self.front_camera_names = tuple(front_camera_names)
        self.back_camera_names = tuple(back_camera_names)
        self.left_camera_names = tuple(left_camera_names)
        self.right_camera_names = tuple(right_camera_names)
        self.pursuer_recognition_tokens = tuple(pursuer_recognition_tokens)

        self.steering_targets = np.array([-0.50, -0.25, 0.0, 0.25, 0.50], dtype=np.float32)
        self.drive_targets = np.array([-0.50, 0.0, 0.35, 0.70, 1.0], dtype=np.float32)
        if self.discrete_actions:
            self.action_space = spaces.MultiDiscrete([self.steering_targets.size, self.drive_targets.size])
        else:
            self.action_space = spaces.Box(
                low=np.array([-0.55, -1.0], dtype=np.float32),
                high=np.array([0.55, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32),
                "vision": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "pursuer": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "ego": spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
                "avoidance": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            }
        )

        self.driver: Any = None
        self.directional_lidars: dict[str, Any] = {}
        self.gps: Any = None
        self.gyro: Any = None
        self.touch_sensor: Any = None
        self.display: Any = None
        self.front_camera: Any = None
        self.back_camera: Any = None
        self.left_camera: Any = None
        self.right_camera: Any = None
        self.evader_node: Any = None
        self.pursuer_node: Any = None
        self.evader_translation_field: Any = None
        self.evader_rotation_field: Any = None
        self.pursuer_translation_field: Any = None
        self.pursuer_rotation_field: Any = None
        self.pursuer_agents: list[dict[str, Any]] = []
        self.random_obstacles: list[tuple[Any, Any, Any | None, float, str, ObstacleFootprint]] = []
        self.pursuer_obstacles: list[tuple[Any, Any, Any | None, ObstacleFootprint, str]] = []
        self.pursuer_obstacle_cache: list[WorldFootprint] = []
        self.obstacle_footprint_cache: dict[str, ObstacleFootprint] = {}

        self.timestep = 32
        self.step_count = 0
        self.previous_distance = 0.0
        self.previous_sector_distances = {"front": 25.0, "left": 25.0, "right": 25.0, "back": 25.0}
        self.previous_position: np.ndarray | None = None
        self.previous_speed_mps = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.current_steering = 0.0
        self.steering_direction_streak = 0
        self.previous_steering_direction = 0
        self.obstacle_turn_direction = 0
        self.obstacle_avoidance_start_heading: float | None = None
        self.raw_action = np.zeros(2, dtype=np.float32)
        self.obstacle_safety_active = False
        self.obstacle_safety_action_delta = 0.0
        self.evader_steering_target = 0.0
        self.evader_steering_command_target = 0.0
        self.evader_steering_stabilization_active = False
        self.evader_steering_stabilization_delta = 0.0
        self.evader_steering_commit_direction = 0
        self.evader_steering_commit_until_step = 0
        self.pursuer_avoidance_active = False
        self.pursuer_avoidance_obstacle_count = 0
        self.pursuer_avoidance_side = 0
        self.pursuer_avoidance_commit_until_step = 0
        self.pursuer_return_to_chase_until_step = 0
        self.pursuer_avoidance_obstacle_center: np.ndarray | None = None
        self.pursuer_avoidance_obstacle_radius = 0.0
        self.pursuer_lidar_avoidance_active = False
        self.pursuer_lidar_front_distance = self.pursuer_lidar_range
        self.pursuer_lidar_left_distance = self.pursuer_lidar_range
        self.pursuer_lidar_right_distance = self.pursuer_lidar_range
        self.pursuer_lidar_danger_count = 0
        self.pursuer_planner_active = False
        self.pursuer_planner_path_length = 0
        self.pursuer_planner_replan = False
        self.pursuer_planner_stuck_recovery = False
        self.pursuer_planner_last_replan_step = -1_000_000
        self.pursuer_direct_chase_hold_until_step = 0
        self.pursuer_planner_path: list[np.ndarray] = []
        self.pursuer_planner_goal_cell: tuple[int, int] | None = None
        self.pursuer_planner_grid: np.ndarray | None = None
        self.pursuer_planner_origin = np.zeros(2, dtype=np.float32)
        self.pursuer_planner_shape = (0, 0)
        self.pursuer_planner_positions: list[np.ndarray] = []
        self.pursuer_line_of_sight = False
        self.pursuer_info_mode = 0
        self.pursuer_hint_refresh = False
        self.pursuer_last_hint_step = -1_000_000
        self.pursuer_last_patrol_step = -1_000_000
        self.pursuer_hint_unit = np.array([1.0, 0.0], dtype=np.float32)
        self.pursuer_search_unit = np.array([1.0, 0.0], dtype=np.float32)
        self.last_reward_log_step = 0
        self.last_timing_log_step = -1_000_000
        self.last_step_timing: dict[str, float] = {}
        self.label_debug_reported = False
        self.directional_lidar_ranges: dict[str, np.ndarray] = {}
        self.recognition_enabled_cameras: set[int] = set()
        self.previous_pursuer_visible = True
        self.previous_pursuer_visual_size = 0.0
        self.visited_exploration_cells: set[tuple[int, int]] = set()

        self.spawn_poses = DEFAULT_SPAWN_POSES

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._ensure_webots()
        self.step_count = 0
        self.current_steering = 0.0
        self.steering_direction_streak = 0
        self.previous_steering_direction = 0
        self.obstacle_turn_direction = 0
        self.obstacle_avoidance_start_heading = None
        self.raw_action = np.zeros(2, dtype=np.float32)
        self.obstacle_safety_active = False
        self.obstacle_safety_action_delta = 0.0
        self.evader_steering_target = 0.0
        self.evader_steering_command_target = 0.0
        self.evader_steering_stabilization_active = False
        self.evader_steering_stabilization_delta = 0.0
        self.evader_steering_commit_direction = 0
        self.evader_steering_commit_until_step = 0
        self.pursuer_avoidance_active = False
        self.pursuer_avoidance_obstacle_count = 0
        self.pursuer_avoidance_side = 0
        self.pursuer_avoidance_commit_until_step = 0
        self.pursuer_return_to_chase_until_step = 0
        self.pursuer_avoidance_obstacle_center = None
        self.pursuer_avoidance_obstacle_radius = 0.0
        self.pursuer_lidar_avoidance_active = False
        self.pursuer_lidar_front_distance = self.pursuer_lidar_range
        self.pursuer_lidar_left_distance = self.pursuer_lidar_range
        self.pursuer_lidar_right_distance = self.pursuer_lidar_range
        self.pursuer_lidar_danger_count = 0
        self.pursuer_planner_active = False
        self.pursuer_planner_path_length = 0
        self.pursuer_planner_replan = False
        self.pursuer_planner_stuck_recovery = False
        self.pursuer_planner_last_replan_step = -1_000_000
        self.pursuer_direct_chase_hold_until_step = 0
        self.pursuer_planner_path = []
        self.pursuer_planner_goal_cell = None
        self.pursuer_planner_positions = []
        self.pursuer_line_of_sight = False
        self.pursuer_info_mode = 0
        self.pursuer_hint_refresh = False
        self.pursuer_last_hint_step = -1_000_000
        self.pursuer_last_patrol_step = -1_000_000
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_speed_mps = 0.0
        self.visited_exploration_cells.clear()

        pose = self._spawn_pose_for_episode()
        evader_spawn_xy = np.array(pose.evader_xy, dtype=np.float32)
        if self.evader_translation_field is not None:
            self._set_vehicle_pose(
                self.evader_node,
                self.evader_translation_field,
                self.evader_rotation_field,
                pose.evader_xy,
                pose.heading,
                z=0.45,
            )
        if self.randomize_obstacles or self.enriched_random_obstacles:
            self._randomize_obstacle_poses()
        self._refresh_pursuer_obstacle_cache()
        self._rebuild_pursuer_planner_grid()

        pursuer_spawn_xy = self._pursuer_spawn_xy_for_episode(evader_spawn_xy, np.array(pose.pursuer_xy, dtype=np.float32))
        pursuer_spawn_heading = self._pursuer_spawn_heading(pose.heading)
        self._set_pursuer_spawn_poses(evader_spawn_xy, pursuer_spawn_xy, pursuer_spawn_heading)
        self._reset_pursuer_limited_info_state()

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(0.0)
        self.driver.setThrottle(0.0)
        self.driver.setBrakeIntensity(1.0)
        self.driver.setSteeringAngle(0.0)
        self.directional_lidar_ranges = {}
        self._step_simulation(self.reset_sensor_warmup_steps)
        obs = self._observation()

        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        self.previous_position = evader_xy
        self.previous_distance = self._distance(evader_xy, pursuer_xy)
        self.visited_exploration_cells.add(self._exploration_cell(evader_xy))
        self.previous_sector_distances = self._lidar_sector_distances()
        initial_vision = self._camera_pursuer_observation()
        self.previous_pursuer_visible = bool(initial_vision["visible"])
        self.previous_pursuer_visual_size = float(initial_vision["visual_size"])
        return obs, {"distance_to_pursuer": self.previous_distance}

    def step(self, action: np.ndarray):
        timing_start = time.perf_counter()
        phase_start = timing_start
        self._ensure_webots()
        self.step_count += 1

        (
            policy_steering,
            policy_drive,
            steering,
            drive,
            policy_steering_index,
            policy_drive_index,
        ) = self._decode_action(action)
        self.raw_action = np.array([steering, drive], dtype=np.float32)
        steering, drive, obstacle_safety_active = self._apply_obstacle_safety(steering, drive)
        safety_action = np.array([steering, drive], dtype=np.float32)
        steering_target = steering
        steering_command_target = self._stabilize_evader_steering(steering_target)
        self.obstacle_safety_active = bool(obstacle_safety_active)
        self.evader_steering_target = float(steering_target)
        self.evader_steering_command_target = float(steering_command_target)
        self.obstacle_safety_action_delta = float(np.linalg.norm(safety_action - self.raw_action))
        throttle, brake, target_speed = self._drive_command_from_action(drive)

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(target_speed)
        self.driver.setThrottle(throttle)
        self.driver.setBrakeIntensity(brake)

        control_elapsed = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        if self.step_count > self.pursuer_start_delay_steps:
            self._move_pursuer()
        pursuer_elapsed = time.perf_counter() - phase_start

        sim_elapsed, max_driver_step_elapsed, driver_step_count = self._step_simulation(
            self.action_repeat,
            steering_target=steering_command_target,
        )

        clipped_action = np.array([self.current_steering, drive], dtype=np.float32)
        self.evader_steering_stabilization_delta = float(abs(self.current_steering - steering_target))
        self.evader_steering_stabilization_active = (
            abs(steering_command_target - steering_target) > 1e-6
            or abs(self.current_steering - steering_command_target) > 1e-6
        )
        self._update_steering_streak(self.current_steering)

        phase_start = time.perf_counter()
        obs = self._observation()
        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        distance = self._distance(evader_xy, pursuer_xy)
        sector_distances = self._lidar_sector_distances()
        obstacle_distance = min(sector_distances.values())
        front_obstacle_distance = sector_distances["front"]
        collision_distance = obstacle_distance
        min_lidar = collision_distance / self._lidar_max_range()
        front_lidar = front_obstacle_distance / self._lidar_max_range()

        moved_distance = self._moved_distance(evader_xy)
        speed_kmh = abs(self._current_speed_kmh())
        observation_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reward, reward_parts = self._reward(
            distance,
            sector_distances,
            clipped_action,
            speed_kmh,
            moved_distance,
            evader_xy,
            pursuer_xy,
        )
        reward_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        captured = distance <= self.capture_distance
        obstacle_collision = self._has_collision(min_lidar, obstacle_distance)
        rollover = self._evader_tilt_angle() >= self.rollover_termination_angle

        if captured:
            reward -= self.capture_termination_penalty
            terminated = True
        if obstacle_collision:
            reward -= self.obstacle_collision_termination_penalty
            terminated = True
        if rollover:
            reward -= self.rollover_termination_penalty
            terminated = True
        if terminated:
            self._stop_vehicle()

        self.previous_distance = distance
        self.previous_sector_distances = sector_distances
        self.previous_position = evader_xy
        self.previous_speed_mps = speed_kmh / 3.6
        self.previous_action = clipped_action
        info = {
            "distance_to_pursuer": distance,
            "min_lidar": min_lidar,
            "front_lidar": front_lidar,
            "obstacle_distance": obstacle_distance,
            "front_obstacle_distance": front_obstacle_distance,
            "left_obstacle_distance": sector_distances["left"],
            "right_obstacle_distance": sector_distances["right"],
            "back_obstacle_distance": sector_distances["back"],
            **reward_parts,
            "captured": captured,
            "obstacle_collision": obstacle_collision,
            "rollover": rollover,
            "touch_contact": self._has_touch_contact(),
            "raw_touch_sensor_contact": self._raw_touch_sensor_contact(),
            "touch_collision_plausible": self._touch_sensor_contact_is_plausible(min_lidar),
            "moved_distance": moved_distance,
            "target_speed": target_speed,
            "throttle": throttle,
            "brake": brake,
            "speed_kmh": speed_kmh,
            "policy_steering": policy_steering,
            "policy_drive": policy_drive,
            "policy_steering_index": float(policy_steering_index),
            "policy_drive_index": float(policy_drive_index),
            "quantized_steering": float(self.raw_action[0]),
            "quantized_drive": float(self.raw_action[1]),
            "applied_steering": float(clipped_action[0]),
            "applied_drive": float(clipped_action[1]),
            "current_steering": float(self.current_steering),
            "steering_direction_streak": float(self.steering_direction_streak),
            "obstacle_turn_direction": float(self.obstacle_turn_direction),
            "obstacle_safety_active": obstacle_safety_active,
            "obstacle_safety_action_delta": self.obstacle_safety_action_delta,
            "evader_steering_target": float(self.evader_steering_target),
            "evader_steering_command_target": float(self.evader_steering_command_target),
            "evader_steering_stabilization_active": float(self.evader_steering_stabilization_active),
            "evader_steering_stabilization_delta": float(self.evader_steering_stabilization_delta),
            "evader_steering_commit_direction": float(self.evader_steering_commit_direction),
            "evader_steering_commit_steps_left": float(max(self.evader_steering_commit_until_step - self.step_count, 0)),
            "pursuer_avoidance_active": float(self.pursuer_avoidance_active),
            "pursuer_avoidance_obstacle_count": float(self.pursuer_avoidance_obstacle_count),
            "pursuer_avoidance_side": float(self.pursuer_avoidance_side),
            "pursuer_avoidance_commit_steps_left": float(max(self.pursuer_avoidance_commit_until_step - self.step_count, 0)),
            "pursuer_return_to_chase_steps_left": float(max(self.pursuer_return_to_chase_until_step - self.step_count, 0)),
            "pursuer_lidar_avoidance_active": float(self.pursuer_lidar_avoidance_active),
            "pursuer_lidar_front_distance": float(self.pursuer_lidar_front_distance),
            "pursuer_lidar_left_distance": float(self.pursuer_lidar_left_distance),
            "pursuer_lidar_right_distance": float(self.pursuer_lidar_right_distance),
            "pursuer_lidar_danger_count": float(self.pursuer_lidar_danger_count),
            "pursuer_planner_active": float(self.pursuer_planner_active),
            "pursuer_planner_path_length": float(self.pursuer_planner_path_length),
            "pursuer_planner_replan": float(self.pursuer_planner_replan),
            "pursuer_planner_stuck_recovery": float(self.pursuer_planner_stuck_recovery),
            "pursuer_count": float(len(self.pursuer_agents) if self.pursuer_agents else 1),
            "pursuer_direct_chase_hold_steps_left": float(
                max(self.pursuer_direct_chase_hold_until_step - self.step_count, 0)
            ),
            "pursuer_behavior_limited": float(self.pursuer_behavior_mode == "limited_info_patrol"),
            "pursuer_line_of_sight": float(self.pursuer_line_of_sight),
            "pursuer_info_mode": float(self.pursuer_info_mode),
            "pursuer_hint_refresh": float(self.pursuer_hint_refresh),
            "pursuer_hint_age_seconds": float(self._pursuer_hint_age_seconds()),
        }
        finish_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.show_reward_display and self.step_count % self.reward_display_interval == 0:
            self._draw_reward_label(reward, info)
            self._draw_supervisor_minimap(info)
            if self.show_car_display:
                self._draw_reward_display(reward, info)
        display_elapsed = time.perf_counter() - phase_start

        timing = {
            "timing_total_ms": (time.perf_counter() - timing_start) * 1000.0,
            "timing_control_ms": control_elapsed * 1000.0,
            "timing_pursuer_ms": pursuer_elapsed * 1000.0,
            "timing_webots_ms": sim_elapsed * 1000.0,
            "timing_webots_max_step_ms": max_driver_step_elapsed * 1000.0,
            "timing_webots_steps": float(driver_step_count),
            "timing_observation_ms": observation_elapsed * 1000.0,
            "timing_reward_ms": reward_elapsed * 1000.0,
            "timing_finish_ms": finish_elapsed * 1000.0,
            "timing_display_ms": display_elapsed * 1000.0,
        }
        self.last_step_timing = timing
        info.update(timing)
        self._maybe_log_slow_step_timing(timing)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.driver is not None:
            self._stop_vehicle()

    def _stop_vehicle(self) -> None:
        if self.driver is None:
            return
        try:
            self.driver.setCruisingSpeed(0.0)
            self.driver.setThrottle(0.0)
            self.driver.setBrakeIntensity(1.0)
        except Exception:
            pass

    def _ensure_webots(self) -> None:
        if self.driver is not None:
            return

        configure_webots_paths()
        os.environ["WEBOTS_ROBOT_NAME"] = self.robot_name
        try:
            from vehicle import Driver
        except ImportError as exc:
            raise RuntimeError(
                "Webots Python vehicle bindings were not found. Run this environment "
                "from a Webots external controller, or set WEBOTS_HOME correctly."
            ) from exc

        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        sensor_timestep = max(int(self.sensor_timestep or self.timestep), self.timestep)
        self.directional_lidars = {
            "front": self._device_by_name(("front lidar",)),
            "left": self._device_by_name(("left lidar",)),
            "right": self._device_by_name(("right lidar",)),
            "back": self._device_by_name(("back lidar",)),
        }
        self.directional_lidars = {name: lidar for name, lidar in self.directional_lidars.items() if lidar is not None}
        self.gps = self._device_by_name(("gps", "GPS"))
        self.gyro = self._device_by_name(("gyro", "Gyro"))
        self.touch_sensor = self._device_by_name(("touch sensor", "evader touch sensor", "TouchSensor"))
        self.display = self._device_by_name(("display", "Display"))
        self.front_camera = self._device_by_name(self.front_camera_names)
        self.back_camera = self._device_by_name(self.back_camera_names)
        self.left_camera = self._device_by_name(self.left_camera_names)
        self.right_camera = self._device_by_name(self.right_camera_names)

        for lidar in self.directional_lidars.values():
            lidar.enable(sensor_timestep)
        if self.enable_camera_recognition:
            for camera in (self.front_camera, self.back_camera, self.left_camera, self.right_camera):
                self._enable_camera(camera, sensor_timestep)
        if self.gps is not None:
            self.gps.enable(sensor_timestep)
        if self.gyro is not None:
            self.gyro.enable(sensor_timestep)
        if self.touch_sensor is not None:
            self.touch_sensor.enable(sensor_timestep)

        self.evader_node = self._node_by_def("Evader")
        self.pursuer_node = self._node_by_def("Pursuer")
        if self.evader_node is not None:
            self.evader_translation_field = self.evader_node.getField("translation")
            self.evader_rotation_field = self.evader_node.getField("rotation")
        if self.pursuer_node is not None:
            self.pursuer_translation_field = self.pursuer_node.getField("translation")
            self.pursuer_rotation_field = self.pursuer_node.getField("rotation")
        self._ensure_pursuer_agents()
        self._collect_random_obstacles()
        self._collect_pursuer_obstacles()

    def _device_by_name(self, names: tuple[str, ...]):
        for name in names:
            try:
                return self.driver.getDevice(name)
            except Exception:
                continue
        return None

    def _node_by_def(self, def_name: str):
        try:
            return self.driver.getFromDef(def_name)
        except Exception:
            return None

    def _root_node(self):
        try:
            return self.driver.getRoot()
        except Exception:
            return None

    def _ensure_pursuer_agents(self) -> None:
        self.pursuer_agents = []
        if self.pursuer_node is not None and self.pursuer_translation_field is not None and self.pursuer_rotation_field is not None:
            self.pursuer_agents.append(
                {
                    "node": self.pursuer_node,
                    "translation_field": self.pursuer_translation_field,
                    "rotation_field": self.pursuer_rotation_field,
                    "index": 0,
                }
            )

        root = self._root_node()
        if root is None:
            return
        children = root.getField("children")
        if children is None:
            return

        for index in range(1, self.pursuer_count):
            def_name = f"Pursuer_{index + 1}"
            node = self._node_by_def(def_name)
            if node is None:
                node_string = (
                    f"DEF {def_name} BmwX5Simple {{ "
                    f"translation 0 0 0.55 "
                    f"rotation 0 0 1 0 "
                    f"name \"pursuer {index + 1}\" "
                    f"color 1 {max(0.05, 0.35 - 0.06 * index):.2f} {max(0.05, 0.12 + 0.08 * index):.2f} "
                    f"}}"
                )
                try:
                    insert_index = children.getCount()
                    children.importMFNodeFromString(-1, node_string)
                    node = self._node_by_def(def_name)
                    if node is None and children.getCount() > insert_index:
                        node = children.getMFNode(insert_index)
                except Exception as exc:
                    print(f"Could not spawn extra pursuer '{def_name}': {exc}")
                    continue
            if node is None:
                print(f"Could not spawn extra pursuer '{def_name}': Webots did not return the inserted node.")
                continue
            translation_field = node.getField("translation")
            rotation_field = node.getField("rotation")
            if translation_field is None or rotation_field is None:
                print(f"Could not use extra pursuer '{def_name}': missing translation or rotation field.")
                continue
            self.pursuer_agents.append(
                {
                    "node": node,
                    "translation_field": translation_field,
                    "rotation_field": rotation_field,
                    "index": index,
                }
            )
        self._park_unused_pursuer_agents()

    def _park_unused_pursuer_agents(self) -> None:
        first_unused_number = self.pursuer_count + 1
        for pursuer_number in range(first_unused_number, first_unused_number + 64):
            node = self._node_by_def(f"Pursuer_{pursuer_number}")
            if node is None:
                continue
            translation_field = node.getField("translation")
            rotation_field = node.getField("rotation")
            if translation_field is not None:
                translation_field.setSFVec3f([0.0, 0.0, -1000.0])
            if rotation_field is not None:
                rotation_field.setSFRotation([0.0, 0.0, 1.0, 0.0])
            if self.reset_vehicle_physics:
                try:
                    node.resetPhysics()
                except Exception:
                    pass

    def _step_simulation(self, substeps: int, steering_target: float | None = None) -> tuple[float, float, int]:
        started = time.perf_counter()
        max_driver_step_elapsed = 0.0
        driver_step_count = 0
        for _ in range(max(int(substeps), 0)):
            if steering_target is not None:
                self._advance_evader_steering(steering_target)
            driver_step_started = time.perf_counter()
            if self.driver.step() == -1:
                max_driver_step_elapsed = max(max_driver_step_elapsed, time.perf_counter() - driver_step_started)
                break
            driver_step_count += 1
            max_driver_step_elapsed = max(max_driver_step_elapsed, time.perf_counter() - driver_step_started)
        return time.perf_counter() - started, max_driver_step_elapsed, driver_step_count

    def _maybe_log_slow_step_timing(self, timing: dict[str, float]) -> None:
        if not self.timing_profile_enabled or self.timing_slow_step_seconds <= 0.0:
            return

        total_ms = float(timing.get("timing_total_ms", 0.0))
        max_driver_ms = float(timing.get("timing_webots_max_step_ms", 0.0))
        threshold_ms = self.timing_slow_step_seconds * 1000.0
        if total_ms < threshold_ms and max_driver_ms < threshold_ms:
            return
        if self.step_count - self.last_timing_log_step < self.timing_log_interval_steps:
            return

        self.last_timing_log_step = self.step_count
        print(
            "SLOW_STEP "
            f"step={self.step_count} total={total_ms:.1f}ms "
            f"control={timing.get('timing_control_ms', 0.0):.1f}ms "
            f"pursuer={timing.get('timing_pursuer_ms', 0.0):.1f}ms "
            f"webots={timing.get('timing_webots_ms', 0.0):.1f}ms "
            f"webots_max={max_driver_ms:.1f}ms/"
            f"{int(timing.get('timing_webots_steps', 0.0))} "
            f"obs={timing.get('timing_observation_ms', 0.0):.1f}ms "
            f"reward={timing.get('timing_reward_ms', 0.0):.1f}ms "
            f"finish={timing.get('timing_finish_ms', 0.0):.1f}ms "
            f"display={timing.get('timing_display_ms', 0.0):.1f}ms"
        )

    def _advance_evader_steering(self, target_steering: float) -> None:
        target_steering = float(np.clip(target_steering, -0.55, 0.55))
        if self.evader_steering_rate_limit <= 0.0:
            self.current_steering = target_steering
        else:
            per_substep_limit = self.evader_steering_rate_limit / max(float(self.action_repeat), 1.0)
            delta = float(np.clip(target_steering - self.current_steering, -per_substep_limit, per_substep_limit))
            self.current_steering = float(np.clip(self.current_steering + delta, -0.55, 0.55))
        try:
            self.driver.setSteeringAngle(self.current_steering)
        except Exception:
            pass

    def _spawn_pose_for_episode(self) -> SpawnPose:
        if self.force_center_spawn or (
            (self.randomize_obstacles or self.enriched_random_obstacles) and self.center_spawn_when_random_obstacles
        ):
            base_pose = self.spawn_poses[0]
            pursuer_offset = (
                base_pose.pursuer_xy[0] - base_pose.evader_xy[0],
                base_pose.pursuer_xy[1] - base_pose.evader_xy[1],
            )
            return SpawnPose((0.0, 0.0), pursuer_offset, base_pose.heading)
        return self.spawn_poses[0]

    def _pursuer_spawn_xy_for_episode(
        self,
        evader_xy: np.ndarray,
        fallback_xy: np.ndarray,
        occupied_xy: tuple[np.ndarray, ...] | list[np.ndarray] = (),
    ) -> np.ndarray:
        fallback_xy = self._clamp_pursuer_spawn_xy(fallback_xy)
        if not self.pursuer_random_spawn and self.pursuer_count <= 1:
            return fallback_xy

        min_x, max_x, min_y, max_y = self._pursuer_spawn_bounds_with_margin()
        best_xy: np.ndarray | None = None
        best_score = -float("inf")
        for _ in range(500):
            xy = np.array(
                [
                    self.np_random.uniform(min_x, max_x),
                    self.np_random.uniform(min_y, max_y),
                ],
                dtype=np.float32,
            )
            if self._distance(xy, evader_xy) < self.pursuer_random_spawn_min_evader_distance:
                continue
            spacing = self._pursuer_spawn_spacing(xy, occupied_xy)
            if spacing < self.pursuer_spawn_ring_radius:
                continue
            clearance = self._pursuer_spawn_clearance(xy)
            score = min(clearance, spacing)
            if score > best_score:
                best_score = score
                best_xy = xy
            if clearance >= self.pursuer_random_spawn_obstacle_margin:
                return xy

        if best_xy is not None:
            return best_xy
        return fallback_xy

    def _pursuer_spawn_bounds_with_margin(self) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = self.pursuer_random_spawn_bounds
        margin = self.pursuer_spawn_wall_margin
        inset_min_x = min_x + margin
        inset_max_x = max_x - margin
        inset_min_y = min_y + margin
        inset_max_y = max_y - margin
        if inset_min_x > inset_max_x:
            midpoint = 0.5 * (min_x + max_x)
            inset_min_x = inset_max_x = midpoint
        if inset_min_y > inset_max_y:
            midpoint = 0.5 * (min_y + max_y)
            inset_min_y = inset_max_y = midpoint
        return inset_min_x, inset_max_x, inset_min_y, inset_max_y

    def _clamp_pursuer_spawn_xy(self, xy: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self._pursuer_spawn_bounds_with_margin()
        return np.array(
            [
                float(np.clip(float(xy[0]), min_x, max_x)),
                float(np.clip(float(xy[1]), min_y, max_y)),
            ],
            dtype=np.float32,
        )

    def _pursuer_spawn_spacing(self, xy: np.ndarray, occupied_xy: tuple[np.ndarray, ...] | list[np.ndarray]) -> float:
        if not occupied_xy:
            return float("inf")
        return min(self._distance(xy, occupied) for occupied in occupied_xy)

    def _pursuer_spawn_clearance(self, xy: np.ndarray) -> float:
        if not self.pursuer_obstacle_cache:
            return float("inf")
        clearance = float("inf")
        for obstacle in self.pursuer_obstacle_cache:
            clearance = min(
                clearance,
                self._point_to_world_footprint_distance(
                    xy,
                    obstacle,
                    precise_margin=self.pursuer_random_spawn_obstacle_margin,
                ),
            )
        return clearance

    def _pursuer_spawn_heading(self, fallback_heading: float) -> float:
        if not self.pursuer_random_spawn:
            return fallback_heading
        return float(self.np_random.uniform(-math.pi, math.pi))

    def _reset_pursuer_limited_info_state(self) -> None:
        pursuer_xy = self._pursuer_xy()
        evader_xy = self._evader_xy()
        direct_unit = self._unit_or_none(evader_xy - pursuer_xy)
        heading_unit = self._pursuer_heading_unit(np.array([1.0, 0.0], dtype=np.float32))
        seed_unit = direct_unit if direct_unit is not None else heading_unit
        self.pursuer_hint_unit = seed_unit.astype(np.float32)
        self.pursuer_search_unit = seed_unit.astype(np.float32)
        self.pursuer_last_hint_step = -self._steps_for_seconds(self.pursuer_limited_info_update_seconds)
        self.pursuer_last_patrol_step = -self._steps_for_seconds(self.pursuer_limited_info_patrol_update_seconds)
        self.pursuer_line_of_sight = False
        self.pursuer_info_mode = 0
        self.pursuer_hint_refresh = False

    def _set_vehicle_pose(
        self,
        node: Any,
        translation_field: Any,
        rotation_field: Any,
        xy: tuple[float, float],
        heading: float,
        z: float,
    ) -> None:
        translation_field.setSFVec3f([xy[0], xy[1], z])
        rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])
        if self.reset_vehicle_physics and node is not None:
            node.resetPhysics()

    def _set_pursuer_spawn_poses(self, evader_xy: np.ndarray, base_xy: np.ndarray, heading: float) -> None:
        agents = self.pursuer_agents or [
            {
                "node": self.pursuer_node,
                "translation_field": self.pursuer_translation_field,
                "rotation_field": self.pursuer_rotation_field,
                "index": 0,
            }
        ]
        count = max(len(agents), 1)
        placed_xy: list[np.ndarray] = []
        for agent in agents:
            translation_field = agent.get("translation_field")
            rotation_field = agent.get("rotation_field")
            if translation_field is None or rotation_field is None:
                continue
            index = int(agent.get("index", 0))
            if self.pursuer_random_spawn or count > 1:
                if index == 0:
                    fallback_xy = base_xy
                else:
                    angle = heading + 2.0 * math.pi * (index - 1) / max(count - 1, 1)
                    offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.pursuer_spawn_ring_radius
                    fallback_xy = base_xy + offset
                xy = self._pursuer_spawn_xy_for_episode(evader_xy, fallback_xy, placed_xy)
            else:
                xy = base_xy
            agent_heading = self._pursuer_spawn_heading(heading)
            self._set_vehicle_pose(
                agent.get("node"),
                translation_field,
                rotation_field,
                (float(xy[0]), float(xy[1])),
                agent_heading,
                z=0.55,
            )
            placed_xy.append(np.asarray(xy, dtype=np.float32))

    def _collect_random_obstacles(self) -> None:
        self.random_obstacles = []
        seen_node_ids: set[int] = set()
        for def_name in self.random_obstacle_def_names:
            node = self._node_by_def(def_name)
            if node is None:
                print(f"Random obstacle DEF '{def_name}' was not found.")
                continue
            self._add_random_obstacle_node(node, seen_node_ids, def_name)

        if self.randomize_all_buildings:
            self._collect_random_building_nodes(seen_node_ids)

    def _collect_pursuer_obstacles(self) -> None:
        self.pursuer_obstacles = []
        self.pursuer_obstacle_cache = []
        seen_node_ids: set[int] = set()
        for node, translation_field, rotation_field, _z, label, footprint in self.random_obstacles:
            if self._is_randomizable_building_node(node):
                self._add_pursuer_obstacle_node(node, translation_field, rotation_field, footprint, label, seen_node_ids)

        root = self._root_node()
        if root is not None:
            self._collect_pursuer_obstacles_from_field(root.getField("children"), seen_node_ids)
        self._refresh_pursuer_obstacle_cache()
        self._rebuild_pursuer_planner_grid()

    def _collect_pursuer_obstacles_from_field(self, field: Any | None, seen_node_ids: set[int]) -> None:
        if field is None:
            return
        try:
            count = field.getCount()
        except Exception:
            return
        for index in range(count):
            try:
                child = field.getMFNode(index)
            except Exception:
                continue
            if self._is_randomizable_building_node(child):
                translation_field = child.getField("translation")
                if translation_field is not None:
                    rotation_field = child.getField("rotation")
                    self._add_pursuer_obstacle_node(
                        child,
                        translation_field,
                        rotation_field,
                        self._obstacle_footprint(child, self._node_type_name(child)),
                        self._node_type_name(child),
                        seen_node_ids,
                    )
            for child_field_name in ("children",):
                try:
                    child_field = child.getField(child_field_name)
                except Exception:
                    child_field = None
                self._collect_pursuer_obstacles_from_field(child_field, seen_node_ids)

    def _add_pursuer_obstacle_node(
        self,
        node: Any,
        translation_field: Any,
        rotation_field: Any | None,
        footprint: ObstacleFootprint,
        label: str,
        seen_node_ids: set[int],
    ) -> None:
        node_id = id(node)
        if node_id in seen_node_ids:
            return
        seen_node_ids.add(node_id)
        self.pursuer_obstacles.append((node, translation_field, rotation_field, footprint, label))

    def _refresh_pursuer_obstacle_cache(self) -> None:
        self.pursuer_obstacle_cache = []
        for _node, translation_field, rotation_field, footprint, label in self.pursuer_obstacles:
            try:
                values = translation_field.getSFVec3f()
            except Exception:
                continue
            xy = np.array([values[0], values[1]], dtype=np.float32)
            if not np.all(np.isfinite(xy)):
                continue
            yaw = self._rotation_field_yaw(rotation_field)
            self.pursuer_obstacle_cache.append(self._world_footprint(xy, yaw, footprint, label))
        self._clear_pursuer_plan()

    def _add_random_obstacle_node(self, node: Any, seen_node_ids: set[int], label: str) -> None:
        node_id = id(node)
        if node_id in seen_node_ids:
            return
        if self._is_excluded_obstacle_node(node, label):
            return
        translation_field = node.getField("translation")
        if translation_field is None:
            print(f"Random obstacle '{label}' has no translation field.")
            return
        rotation_field = node.getField("rotation")
        try:
            z = float(translation_field.getSFVec3f()[2])
        except Exception:
            z = 0.0
        seen_node_ids.add(node_id)
        footprint = self._obstacle_footprint(node, label)
        self.random_obstacles.append((node, translation_field, rotation_field, z, label, footprint))

    def _collect_random_building_nodes(self, seen_node_ids: set[int]) -> None:
        root = self._root_node()
        if root is None:
            return
        self._collect_random_building_nodes_from_field(root.getField("children"), seen_node_ids)

    def _collect_random_building_nodes_from_field(self, field: Any | None, seen_node_ids: set[int]) -> None:
        if field is None:
            return
        try:
            count = field.getCount()
        except Exception:
            return
        for index in range(count):
            try:
                child = field.getMFNode(index)
            except Exception:
                continue
            if self._is_randomizable_building_node(child):
                self._add_random_obstacle_node(child, seen_node_ids, self._node_type_name(child))
            for child_field_name in ("children", "signBoards", "rightHorizontalSigns", "rightVerticalSigns", "leftVerticalSigns"):
                try:
                    child_field = child.getField(child_field_name)
                except Exception:
                    child_field = None
                self._collect_random_building_nodes_from_field(child_field, seen_node_ids)

    def _is_randomizable_building_node(self, node: Any) -> bool:
        if self._is_excluded_obstacle_node(node):
            return False
        return self._node_type_name(node) in {
            "BuildingUnderConstruction",
            "CommercialBuilding",
            "UBuilding",
            "HollowBuilding",
            "Hotel",
            "TheThreeTowers",
            "CyberboticsTower",
            "BigGlassTower",
            "Auditorium",
            "Museum",
            "ResidentialBuilding",
            "FastFoodRestaurant",
            "SimpleBuilding",
        }

    def _is_excluded_obstacle_node(self, node: Any, label: str = "") -> bool:
        type_name = self._node_type_name(node)
        return type_name in self.obstacle_excluded_type_names or label in self.obstacle_excluded_def_names

    def _node_type_name(self, node: Any) -> str:
        for method_name in ("getTypeName", "get_type_name"):
            method = getattr(node, method_name, None)
            if method is None:
                continue
            try:
                return str(method())
            except Exception:
                continue
        return ""

    def _obstacle_footprint(self, node: Any, label: str) -> ObstacleFootprint:
        cache_key = self._obstacle_footprint_cache_key(node, label)
        cached = self.obstacle_footprint_cache.get(cache_key)
        if cached is not None:
            return cached

        footprint = self._build_obstacle_footprint(node, label)
        self.obstacle_footprint_cache[cache_key] = footprint
        return footprint

    def _obstacle_footprint_cache_key(self, node: Any, label: str) -> str:
        type_name = self._node_type_name(node)

        length = self._float_field_value(node, "length")
        width = self._float_field_value(node, "width")
        if length is not None and width is not None:
            return f"{type_name}:rect:{length:.3f}:{width:.3f}"

        return f"{type_name or label}:default"

    @staticmethod
    def _vertices_cache_key(vertices: tuple[tuple[float, float], ...]) -> str:
        return ";".join(f"{x:.3f},{y:.3f}" for x, y in vertices)

    def _build_obstacle_footprint(self, node: Any, label: str) -> ObstacleFootprint:
        type_name = self._node_type_name(node)

        length = self._float_field_value(node, "length")
        width = self._float_field_value(node, "width")
        if length is not None and width is not None:
            return ObstacleFootprint(0.5 * math.hypot(max(float(length), 0.1), max(float(width), 0.1)))

        override = self._building_footprint_override(type_name)
        if override is not None:
            return override

        if type_name == "TrafficCone":
            return ObstacleFootprint(0.8)
        if label == "STONES":
            return ObstacleFootprint(15.0)
        if type_name == "SimpleBuilding":
            return ObstacleFootprint(12.0)
        building_radius = self._building_radius_fallback(type_name)
        if building_radius is not None:
            return ObstacleFootprint(building_radius)
        return ObstacleFootprint(6.0)

    @staticmethod
    def _building_footprint_override(type_name: str) -> ObstacleFootprint | None:
        radii = {
            "TheThreeTowers": 30.0,
            "Auditorium": 30.0,
        }
        radius = radii.get(type_name)
        if radius is None:
            return None
        return ObstacleFootprint(radius)

    def _bounding_object_footprint(self, node: Any) -> ObstacleFootprint | None:
        bounding_node = self._sf_node_field(node, "boundingObject", base=True)
        if bounding_node is None:
            bounding_node = self._sf_node_field(node, "boundingObject", base=False)
        if bounding_node is None:
            return None

        parts = self._footprint_parts_from_node(bounding_node, np.eye(3, dtype=np.float32), depth=0)
        if not parts:
            return None
        return self._parts_footprint(parts)

    def _footprint_parts_from_node(
        self,
        node: Any,
        transform: np.ndarray,
        depth: int,
    ) -> list[tuple[tuple[float, float], ...]]:
        if node is None or depth > 12:
            return []

        type_name = self._node_type_name(node)
        base_type_name = self._node_base_type_name(node)
        names = {type_name, base_type_name}

        local_transform = self._node_local_2d_transform(node)
        combined_transform = transform @ local_transform
        parts: list[tuple[tuple[float, float], ...]] = []

        if "Box" in names:
            size = self._vec3_field_value(node, "size")
            if size is not None:
                parts.append(self._transform_vertices(self._box_vertices(float(size[0]), float(size[1])), combined_transform))
        elif "Cylinder" in names:
            radius = self._float_field_value(node, "radius")
            if radius is not None:
                parts.append(
                    self._transform_vertices(
                        self._circle_vertices(float(radius), segments=16),
                        combined_transform,
                    )
                )
        elif "Sphere" in names:
            radius = self._float_field_value(node, "radius")
            if radius is not None:
                parts.append(
                    self._transform_vertices(
                        self._circle_vertices(float(radius), segments=16),
                        combined_transform,
                    )
                )
        elif "Plane" in names:
            return []
        elif "IndexedFaceSet" in names:
            indexed_vertices = self._indexed_face_set_vertices(node)
            if indexed_vertices:
                parts.append(self._transform_vertices(indexed_vertices, combined_transform))

        geometry = self._sf_node_field(node, "geometry")
        if geometry is not None:
            parts.extend(self._footprint_parts_from_node(geometry, combined_transform, depth + 1))

        for field_name in ("children", "boundingObject"):
            children = self._mf_node_field(node, field_name)
            for child in children:
                parts.extend(self._footprint_parts_from_node(child, combined_transform, depth + 1))

        return parts

    def _parts_footprint(self, parts: list[tuple[tuple[float, float], ...]]) -> ObstacleFootprint | None:
        clean_parts = tuple(part for part in parts if len(part) >= 3)
        if not clean_parts:
            return None
        radius = 0.0
        for part in clean_parts:
            for x, y in part:
                radius = max(radius, math.hypot(x, y))
        vertices = clean_parts[0] if len(clean_parts) == 1 else ()
        return ObstacleFootprint(max(radius, 0.1), vertices, clean_parts)

    def _node_base_type_name(self, node: Any) -> str:
        for method_name in ("getBaseTypeName", "get_base_type_name"):
            method = getattr(node, method_name, None)
            if method is None:
                continue
            try:
                return str(method())
            except Exception:
                continue
        return ""

    def _node_local_2d_transform(self, node: Any) -> np.ndarray:
        translation = self._vec3_field_value(node, "translation") or (0.0, 0.0, 0.0)
        rotation = self._rotation_field_value(node, "rotation")
        yaw = 0.0 if rotation is None else self._rotation_yaw(rotation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return np.array(
            [
                [cos_yaw, -sin_yaw, float(translation[0])],
                [sin_yaw, cos_yaw, float(translation[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _box_vertices(length: float, width: float) -> tuple[tuple[float, float], ...]:
        half_length = max(float(length), 0.1) * 0.5
        half_width = max(float(width), 0.1) * 0.5
        return (
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        )

    @staticmethod
    def _circle_vertices(radius: float, segments: int = 16) -> tuple[tuple[float, float], ...]:
        radius = max(float(radius), 0.1)
        segments = max(int(segments), 8)
        return tuple(
            (
                radius * math.cos(2.0 * math.pi * index / segments),
                radius * math.sin(2.0 * math.pi * index / segments),
            )
            for index in range(segments)
        )

    @staticmethod
    def _transform_vertices(
        vertices: tuple[tuple[float, float], ...],
        transform: np.ndarray,
    ) -> tuple[tuple[float, float], ...]:
        transformed: list[tuple[float, float]] = []
        for x, y in vertices:
            vector = np.array([x, y, 1.0], dtype=np.float32)
            result = transform @ vector
            transformed.append((float(result[0]), float(result[1])))
        return tuple(transformed)

    def _indexed_face_set_vertices(self, node: Any) -> tuple[tuple[float, float], ...]:
        coord_node = self._sf_node_field(node, "coord")
        if coord_node is None:
            return ()
        point_field = self._field_by_name(coord_node, "point")
        if point_field is None:
            return ()
        points: list[tuple[float, float]] = []
        try:
            count = point_field.getCount()
        except Exception:
            return ()
        for index in range(count):
            point = self._mf_vec(point_field, index)
            if point is None or len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))
        if len(points) < 3:
            return ()
        return self._convex_hull(points)

    @staticmethod
    def _convex_hull(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        unique_points = sorted(set(points))
        if len(unique_points) <= 1:
            return tuple(unique_points)

        def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
            return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

        lower: list[tuple[float, float]] = []
        for point in unique_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)

        upper: list[tuple[float, float]] = []
        for point in reversed(unique_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)

        return tuple(lower[:-1] + upper[:-1])

    def _sf_node_field(self, node: Any, field_name: str, base: bool = False) -> Any | None:
        field = self._field_by_name(node, field_name, base=base)
        if field is None:
            return None
        for method_name in ("getSFNode", "get_sf_node"):
            method = getattr(field, method_name, None)
            if method is None:
                continue
            try:
                return method()
            except Exception:
                continue
        return None

    def _mf_node_field(self, node: Any, field_name: str) -> list[Any]:
        field = self._field_by_name(node, field_name)
        if field is None:
            return []
        try:
            count = field.getCount()
        except Exception:
            return []
        children: list[Any] = []
        for index in range(count):
            try:
                children.append(field.getMFNode(index))
            except Exception:
                continue
        return children

    def _field_by_name(self, node: Any, field_name: str, base: bool = False) -> Any | None:
        method_names = ("getBaseNodeField", "get_base_node_field") if base else ("getField", "get_field")
        for method_name in method_names:
            method = getattr(node, method_name, None)
            if method is None:
                continue
            try:
                return method(field_name)
            except Exception:
                continue
        return None

    @staticmethod
    def _building_radius_fallback(type_name: str) -> float | None:
        return {
            "Museum": 28.0,
            "UBuilding": 26.0,
            "HollowBuilding": 26.0,
            "CommercialBuilding": 22.0,
            "BuildingUnderConstruction": 18.0,
            "Hotel": 22.0,
            "CyberboticsTower": 22.0,
            "BigGlassTower": 22.0,
            "ResidentialBuilding": 18.0,
            "FastFoodRestaurant": 10.0,
        }.get(type_name)

    @staticmethod
    def _rectangle_footprint(length: float, width: float) -> ObstacleFootprint:
        half_length = max(float(length), 0.1) * 0.5
        half_width = max(float(width), 0.1) * 0.5
        return ObstacleFootprint(
            math.hypot(half_length, half_width),
            (
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width),
            ),
        )

    @staticmethod
    def _polygon_footprint(vertices: tuple[tuple[float, float], ...]) -> ObstacleFootprint:
        radius = max(math.hypot(x, y) for x, y in vertices)
        return ObstacleFootprint(max(radius, 0.1), vertices)

    def _corners_footprint_vertices(self, node: Any) -> tuple[tuple[float, float], ...]:
        try:
            corners_field = node.getField("corners")
        except Exception:
            corners_field = None
        if corners_field is None:
            return ()
        try:
            count = corners_field.getCount()
        except Exception:
            return ()

        vertices: list[tuple[float, float]] = []
        for index in range(count):
            corner = None
            for method_name in ("getMFVec2f", "getMFVec3f"):
                method = getattr(corners_field, method_name, None)
                if method is None:
                    continue
                try:
                    corner = method(index)
                    break
                except Exception:
                    continue
            if corner is None or len(corner) < 2:
                continue
            vertices.append((float(corner[0]), float(corner[1])))
        return tuple(vertices) if len(vertices) >= 3 else ()

    def _float_field_value(self, node: Any, field_name: str) -> float | None:
        field = self._field_by_name(node, field_name)
        if field is None:
            return None
        for method_name in ("getSFFloat", "getSFInt32"):
            method = getattr(field, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return None

    def _vec3_field_value(self, node: Any, field_name: str) -> tuple[float, float, float] | None:
        field = self._field_by_name(node, field_name)
        if field is None:
            return None
        for method_name in ("getSFVec3f", "get_sf_vec3f"):
            method = getattr(field, method_name, None)
            if method is None:
                continue
            try:
                value = method()
                if value is not None and len(value) >= 3:
                    return float(value[0]), float(value[1]), float(value[2])
            except Exception:
                continue
        return None

    def _rotation_field_value(self, node: Any, field_name: str) -> tuple[float, float, float, float] | None:
        field = self._field_by_name(node, field_name)
        if field is None:
            return None
        for method_name in ("getSFRotation", "get_sf_rotation"):
            method = getattr(field, method_name, None)
            if method is None:
                continue
            try:
                value = method()
                if value is not None and len(value) >= 4:
                    return float(value[0]), float(value[1]), float(value[2]), float(value[3])
            except Exception:
                continue
        return None

    def _mf_vec(self, field: Any, index: int) -> tuple[float, ...] | None:
        for method_name in ("getMFVec3f", "getMFVec2f", "get_mf_vec3f", "get_mf_vec2f"):
            method = getattr(field, method_name, None)
            if method is None:
                continue
            try:
                value = method(index)
                if value is not None and len(value) >= 2:
                    return tuple(float(component) for component in value)
            except Exception:
                continue
        return None

    @staticmethod
    def _rotation_yaw(rotation: tuple[float, float, float, float] | list[float]) -> float:
        if len(rotation) < 4:
            return 0.0
        axis_z = float(rotation[2])
        angle = float(rotation[3])
        return angle if axis_z >= 0.0 else -angle

    def _randomize_obstacle_poses(self) -> None:
        if not self.random_obstacles:
            return

        placed: list[WorldFootprint] = []
        obstacles = sorted(self.random_obstacles, key=lambda obstacle: obstacle[5].radius, reverse=True)
        for node, translation_field, rotation_field, z, label, footprint in obstacles:
            yaw = float(self.np_random.uniform(-math.pi, math.pi))
            xy = self._sample_random_obstacle_xy(placed, footprint, yaw)
            placed.append(self._world_footprint(xy, yaw, footprint, label))
            self._set_random_obstacle_pose(node, translation_field, rotation_field, z, xy, yaw)

        if self.enriched_random_obstacles:
            self._place_enriched_random_blockers()

    def _set_random_obstacle_pose(
        self,
        node: Any,
        translation_field: Any,
        rotation_field: Any | None,
        z: float,
        xy: np.ndarray,
        yaw: float,
    ) -> None:
        translation_field.setSFVec3f([float(xy[0]), float(xy[1]), z])
        if rotation_field is not None:
            rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])
        try:
            node.resetPhysics()
        except Exception:
            pass

    def _place_enriched_random_blockers(self) -> None:
        candidates = self._enriched_random_candidates()
        if len(candidates) < 2:
            return

        pattern = self._sample_enriched_random_pattern()
        slots = self._enriched_pattern_slots(pattern)
        if len(slots) < 2:
            return

        indices = self.np_random.choice(len(candidates), size=2, replace=False)
        evader_xy = self._evader_xy()
        heading = self._evader_heading()
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        left = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float32)

        for candidate_index, slot in zip(indices, slots):
            node, translation_field, rotation_field, z, _label, footprint = candidates[int(candidate_index)]
            world_xy = self._enriched_slot_position(evader_xy, forward, left, slot, footprint.radius)
            yaw = heading + float(self.np_random.uniform(-0.35, 0.35))
            self._set_random_obstacle_pose(node, translation_field, rotation_field, z, world_xy, yaw)

    def _enriched_random_candidates(self) -> list[tuple[Any, Any, Any | None, float, str, ObstacleFootprint]]:
        if not self.enriched_random_obstacle_def_names:
            return list(self.random_obstacles)

        preferred = set(self.enriched_random_obstacle_def_names)
        candidates = [obstacle for obstacle in self.random_obstacles if obstacle[4] in preferred]
        return candidates if len(candidates) >= 2 else list(self.random_obstacles)

    def _sample_enriched_random_pattern(self) -> str:
        patterns = tuple(pattern for pattern in self.enriched_random_patterns if pattern in {"front_right", "front_left", "left_right"})
        if not patterns:
            return "front_right"
        return str(patterns[int(self.np_random.integers(0, len(patterns)))])

    def _enriched_pattern_slots(self, pattern: str) -> tuple[str, str]:
        if pattern == "front_left":
            return "front", "left"
        if pattern == "left_right":
            return "left", "right"
        return "front", "right"

    def _enriched_slot_position(
        self,
        evader_xy: np.ndarray,
        forward: np.ndarray,
        left: np.ndarray,
        slot: str,
        footprint_radius: float,
    ) -> np.ndarray:
        clearance = self.enriched_random_evader_clearance + max(0.0, footprint_radius)
        jitter = float(self.np_random.uniform(-self.enriched_random_jitter, self.enriched_random_jitter))
        if slot == "front":
            forward_distance = self.enriched_random_front_distance + clearance
            return evader_xy + forward * forward_distance + left * jitter

        side_sign = 1.0 if slot == "left" else -1.0
        side_distance = self.enriched_random_side_distance + clearance
        forward_jitter = float(self.np_random.uniform(-self.enriched_random_jitter, self.enriched_random_jitter))
        return evader_xy + forward * (2.0 + forward_jitter) + left * (side_sign * side_distance)

    def _sample_random_obstacle_xy(
        self,
        placed: list[WorldFootprint],
        footprint: ObstacleFootprint,
        yaw: float,
    ) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.random_obstacle_bounds
        edge_margin = footprint.radius + self.random_obstacle_edge_spacing
        sample_min_x, sample_max_x = self._shrunk_axis_bounds(min_x, max_x, edge_margin)
        sample_min_y, sample_max_y = self._shrunk_axis_bounds(min_y, max_y, edge_margin)
        best_xy: np.ndarray | None = None
        best_score = -float("inf")
        for _ in range(500):
            xy = np.array(
                [
                    self.np_random.uniform(sample_min_x, sample_max_x),
                    self.np_random.uniform(sample_min_y, sample_max_y),
                ],
                dtype=np.float32,
            )
            candidate = self._world_footprint(xy, yaw, footprint, "")
            score = self._random_obstacle_placement_score(candidate, placed)
            if score > best_score:
                best_score = score
                best_xy = xy
            if score < 0.0:
                continue
            return xy

        if best_xy is not None:
            return best_xy
        return np.array([sample_min_x, sample_min_y], dtype=np.float32)

    def _random_obstacle_placement_score(self, candidate: WorldFootprint, placed: list[WorldFootprint]) -> float:
        spawn_clearance = self._point_to_world_footprint_distance(
            self.random_obstacle_exclusion_center,
            candidate,
            precise_margin=self.random_obstacle_exclusion_radius + self.random_obstacle_edge_spacing,
        ) - self.random_obstacle_exclusion_radius - self.random_obstacle_edge_spacing
        score = spawn_clearance
        for other in placed:
            center_clearance = float(np.linalg.norm(candidate.center - other.center)) - self.random_obstacle_min_spacing
            body_clearance = (
                self._world_footprint_distance(
                    candidate,
                    other,
                    precise_margin=self.random_obstacle_edge_spacing,
                )
                - self.random_obstacle_edge_spacing
            )
            score = min(score, center_clearance, body_clearance)
        return score

    @staticmethod
    def _shrunk_axis_bounds(min_value: float, max_value: float, margin: float) -> tuple[float, float]:
        if max_value - min_value <= 2.0 * margin:
            return min_value, max_value
        return min_value + margin, max_value - margin

    def _world_footprint(self, center: np.ndarray, yaw: float, footprint: ObstacleFootprint, label: str) -> WorldFootprint:
        return WorldFootprint(
            center=np.asarray(center, dtype=np.float32),
            radius=footprint.radius,
            label=label,
            vertices=None,
            parts=(),
        )

    @staticmethod
    def _world_footprint_vertices(center: np.ndarray, yaw: float, footprint: ObstacleFootprint) -> np.ndarray | None:
        if not footprint.vertices:
            return None
        local = np.asarray(footprint.vertices, dtype=np.float32)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        return local @ rotation.T + np.asarray(center, dtype=np.float32)

    @staticmethod
    def _world_footprint_parts(center: np.ndarray, yaw: float, footprint: ObstacleFootprint) -> tuple[np.ndarray, ...]:
        source_parts = footprint.parts
        if not source_parts and footprint.vertices:
            source_parts = (footprint.vertices,)
        if not source_parts:
            return ()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        center_array = np.asarray(center, dtype=np.float32)
        return tuple(np.asarray(part, dtype=np.float32) @ rotation.T + center_array for part in source_parts)

    def _point_to_world_footprint_distance(
        self,
        point: np.ndarray,
        obstacle: WorldFootprint,
        precise_margin: float = 0.0,
    ) -> float:
        point = np.asarray(point, dtype=np.float32)
        return max(0.0, float(np.linalg.norm(point - obstacle.center)) - obstacle.radius)

    def _world_footprint_distance(
        self,
        first: WorldFootprint,
        second: WorldFootprint,
        precise_margin: float = 0.0,
    ) -> float:
        return max(0.0, float(np.linalg.norm(first.center - second.center)) - first.radius - second.radius)

    def _segment_to_world_footprint_distance(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacle: WorldFootprint,
        precise_margin: float = 0.0,
    ) -> float:
        return max(0.0, self._point_segment_distance(obstacle.center, start, end) - obstacle.radius)

    def _point_to_polygon_distance(self, point: np.ndarray, polygon: np.ndarray) -> float:
        if self._point_in_polygon(point, polygon):
            return 0.0
        min_distance = float("inf")
        for start, end in self._polygon_edges(polygon):
            min_distance = min(min_distance, self._point_segment_distance(point, start, end))
        return min_distance

    def _polygon_distance(self, first: np.ndarray, second: np.ndarray) -> float:
        if self._polygons_intersect(first, second):
            return 0.0
        min_distance = float("inf")
        for vertex in first:
            min_distance = min(min_distance, self._point_to_polygon_distance(vertex, second))
        for vertex in second:
            min_distance = min(min_distance, self._point_to_polygon_distance(vertex, first))
        return min_distance

    def _segment_to_polygon_distance(self, start: np.ndarray, end: np.ndarray, polygon: np.ndarray) -> float:
        if self._point_in_polygon(start, polygon) or self._point_in_polygon(end, polygon):
            return 0.0
        for edge_start, edge_end in self._polygon_edges(polygon):
            if self._segments_intersect(start, end, edge_start, edge_end):
                return 0.0
        min_distance = min(
            self._point_to_polygon_distance(start, polygon),
            self._point_to_polygon_distance(end, polygon),
        )
        for edge_start, edge_end in self._polygon_edges(polygon):
            min_distance = min(min_distance, self._segment_distance(start, end, edge_start, edge_end))
        return min_distance

    @staticmethod
    def _polygon_edges(polygon: np.ndarray):
        for index in range(len(polygon)):
            yield polygon[index], polygon[(index + 1) % len(polygon)]

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        x = float(point[0])
        y = float(point[1])
        inside = False
        previous = polygon[-1]
        for current in polygon:
            xi, yi = float(current[0]), float(current[1])
            xj, yj = float(previous[0]), float(previous[1])
            crosses = (yi > y) != (yj > y)
            if crosses:
                denominator = yj - yi
                if abs(denominator) <= 1e-12:
                    previous = current
                    continue
                x_at_y = (xj - xi) * (y - yi) / denominator + xi
                if x < x_at_y:
                    inside = not inside
            previous = current
        return inside

    def _polygons_intersect(self, first: np.ndarray, second: np.ndarray) -> bool:
        if any(self._point_in_polygon(vertex, second) for vertex in first):
            return True
        if any(self._point_in_polygon(vertex, first) for vertex in second):
            return True
        for first_start, first_end in self._polygon_edges(first):
            for second_start, second_end in self._polygon_edges(second):
                if self._segments_intersect(first_start, first_end, second_start, second_end):
                    return True
        return False

    @staticmethod
    def _segments_intersect(first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray) -> bool:
        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
            return (
                min(float(a[0]), float(c[0])) - 1e-9 <= float(b[0]) <= max(float(a[0]), float(c[0])) + 1e-9
                and min(float(a[1]), float(c[1])) - 1e-9 <= float(b[1]) <= max(float(a[1]), float(c[1])) + 1e-9
            )

        o1 = orientation(first_start, first_end, second_start)
        o2 = orientation(first_start, first_end, second_end)
        o3 = orientation(second_start, second_end, first_start)
        o4 = orientation(second_start, second_end, first_end)
        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return True
        if abs(o1) <= 1e-9 and on_segment(first_start, second_start, first_end):
            return True
        if abs(o2) <= 1e-9 and on_segment(first_start, second_end, first_end):
            return True
        if abs(o3) <= 1e-9 and on_segment(second_start, first_start, second_end):
            return True
        if abs(o4) <= 1e-9 and on_segment(second_start, first_end, second_end):
            return True
        return False

    def _segment_distance(self, first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray) -> float:
        if self._segments_intersect(first_start, first_end, second_start, second_end):
            return 0.0
        return min(
            self._point_segment_distance(first_start, second_start, second_end),
            self._point_segment_distance(first_end, second_start, second_end),
            self._point_segment_distance(second_start, first_start, first_end),
            self._point_segment_distance(second_end, first_start, first_end),
        )

    def _rotation_field_yaw(self, rotation_field: Any | None) -> float:
        if rotation_field is None:
            return 0.0
        try:
            rotation = rotation_field.getSFRotation()
        except Exception:
            return 0.0
        axis_z = float(rotation[2])
        angle = float(rotation[3])
        return angle if axis_z >= 0.0 else -angle

    def _move_pursuer(self) -> None:
        if not self.pursuer_agents and (self.pursuer_translation_field is None or self.pursuer_rotation_field is None):
            return
        evader_xy = self._evader_xy()
        agents = self.pursuer_agents or [
            {
                "node": self.pursuer_node,
                "translation_field": self.pursuer_translation_field,
                "rotation_field": self.pursuer_rotation_field,
                "index": 0,
            }
        ]
        for agent in agents:
            self._move_single_pursuer(agent, evader_xy)

    def _move_single_pursuer(self, agent: dict[str, Any], evader_xy: np.ndarray) -> None:
        translation_field = agent.get("translation_field")
        rotation_field = agent.get("rotation_field")
        if translation_field is None or rotation_field is None:
            return

        pursuer_xy = self._pursuer_agent_xy(agent)
        is_primary = int(agent.get("index", 0)) == 0
        guidance_unit, knows_exact_position = self._pursuer_guidance_unit(pursuer_xy, evader_xy)
        if guidance_unit is None:
            return

        dt = (self.timestep * self.action_repeat) / 1000.0
        step = self.pursuer_speed_mps * dt
        if knows_exact_position:
            step = min(self._distance(pursuer_xy, evader_xy), step)

        planned = self._pursuer_planner_step(pursuer_xy, evader_xy, guidance_unit, step) if is_primary else None
        if planned is not None:
            next_xy, unit = planned
        elif is_primary:
            next_xy, unit = self._pursuer_collision_avoidance_step(pursuer_xy, evader_xy, guidance_unit, step)
        else:
            next_xy, unit = self._pursuer_simple_collision_avoidance_step(pursuer_xy, evader_xy, guidance_unit, step)

        next_xy, unit = self._pursuer_return_to_chase_step(pursuer_xy, evader_xy, guidance_unit, next_xy, unit, step)
        next_xy = self._constrain_pursuer_xy_to_city(pursuer_xy, next_xy)
        movement = next_xy - pursuer_xy
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm > 1e-6:
            unit = movement / movement_norm
        else:
            unit = self._pursuer_agent_heading_unit(agent, guidance_unit)
        heading = math.atan2(float(unit[1]), float(unit[0]))
        current = translation_field.getSFVec3f()
        translation_field.setSFVec3f([float(next_xy[0]), float(next_xy[1]), current[2]])
        rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])

    def _pursuer_return_to_chase_step(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        direct_unit: np.ndarray,
        proposed_xy: np.ndarray,
        proposed_unit: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.step_count > self.pursuer_return_to_chase_until_step:
            return proposed_xy, proposed_unit
        if not self.pursuer_line_of_sight:
            return proposed_xy, proposed_unit

        direct_next = pursuer_xy + direct_unit * step
        margin = self._pursuer_active_avoidance_margin()
        immediate_hit = self._pursuer_path_building_hit(pursuer_xy, direct_next, evader_xy, margin=margin)
        if immediate_hit is None:
            self.pursuer_return_to_chase_until_step = 0
            return direct_next, direct_unit

        recovery_unit = self._unit_or_none(0.75 * direct_unit + 0.25 * proposed_unit)
        if recovery_unit is None:
            return proposed_xy, proposed_unit
        recovery_next = pursuer_xy + recovery_unit * step
        recovery_hit = self._pursuer_path_building_hit(pursuer_xy, recovery_next, evader_xy, margin=margin)
        if recovery_hit is None:
            return recovery_next, recovery_unit
        return proposed_xy, proposed_unit

    def _constrain_pursuer_xy_to_city(self, current_xy: np.ndarray, next_xy: np.ndarray) -> np.ndarray:
        constrained = self._clamp_pursuer_spawn_xy(next_xy)
        if np.allclose(constrained, next_xy, atol=1e-6):
            return next_xy

        slide_x = np.array([float(constrained[0]), float(current_xy[1])], dtype=np.float32)
        slide_y = np.array([float(current_xy[0]), float(constrained[1])], dtype=np.float32)
        candidates = [constrained, slide_x, slide_y, current_xy]
        best = current_xy
        best_progress = -float("inf")
        intended = next_xy - current_xy
        for candidate in candidates:
            candidate = self._clamp_pursuer_spawn_xy(candidate)
            progress = float(np.dot(candidate - current_xy, intended))
            if progress > best_progress:
                best_progress = progress
                best = candidate
        return best

    def _pursuer_heading_unit(self, fallback: np.ndarray) -> np.ndarray:
        if self.pursuer_rotation_field is None:
            return fallback
        return self._heading_unit_from_rotation_field(self.pursuer_rotation_field, fallback)

    def _pursuer_agent_heading_unit(self, agent: dict[str, Any], fallback: np.ndarray) -> np.ndarray:
        return self._heading_unit_from_rotation_field(agent.get("rotation_field"), fallback)

    def _heading_unit_from_rotation_field(self, rotation_field: Any | None, fallback: np.ndarray) -> np.ndarray:
        if rotation_field is None:
            return fallback
        try:
            rotation = rotation_field.getSFRotation()
        except Exception:
            return fallback
        axis_z = float(rotation[2])
        angle = float(rotation[3])
        heading = angle if axis_z >= 0.0 else -angle
        return np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)

    def _pursuer_guidance_unit(self, pursuer_xy: np.ndarray, evader_xy: np.ndarray) -> tuple[np.ndarray | None, bool]:
        direct_unit = self._unit_or_none(evader_xy - pursuer_xy)
        if direct_unit is None:
            return None, True

        if self.pursuer_behavior_mode != "limited_info_patrol":
            self.pursuer_line_of_sight = True
            self.pursuer_info_mode = 0
            self.pursuer_hint_refresh = False
            return direct_unit, True

        self.pursuer_line_of_sight = self._pursuer_has_line_of_sight(pursuer_xy, evader_xy)
        if self.pursuer_line_of_sight:
            self.pursuer_hint_unit = direct_unit.astype(np.float32)
            self.pursuer_search_unit = direct_unit.astype(np.float32)
            self.pursuer_last_hint_step = self.step_count
            self.pursuer_info_mode = 1
            self.pursuer_hint_refresh = False
            return direct_unit, True

        hint_interval = self._steps_for_seconds(self.pursuer_limited_info_update_seconds)
        if self.step_count - self.pursuer_last_hint_step >= hint_interval:
            noise = math.radians(
                float(
                    self.np_random.uniform(
                        -self.pursuer_limited_info_direction_noise_degrees,
                        self.pursuer_limited_info_direction_noise_degrees,
                    )
                )
            )
            self.pursuer_hint_unit = self._rotate_unit(direct_unit, noise)
            self.pursuer_search_unit = self.pursuer_hint_unit.copy()
            self.pursuer_last_hint_step = self.step_count
            self.pursuer_last_patrol_step = self.step_count
            self.pursuer_info_mode = 2
            self.pursuer_hint_refresh = True
            return self.pursuer_search_unit, False

        patrol_interval = self._steps_for_seconds(self.pursuer_limited_info_patrol_update_seconds)
        if self.step_count - self.pursuer_last_patrol_step >= patrol_interval:
            turn = math.radians(
                float(
                    self.np_random.uniform(
                        -self.pursuer_limited_info_patrol_turn_degrees,
                        self.pursuer_limited_info_patrol_turn_degrees,
                    )
                )
            )
            patrol_unit = self._rotate_unit(self.pursuer_search_unit, turn)
            blended = self._unit_or_none(0.85 * patrol_unit + 0.15 * self.pursuer_hint_unit)
            self.pursuer_search_unit = patrol_unit if blended is None else blended.astype(np.float32)
            self.pursuer_last_patrol_step = self.step_count

        self.pursuer_info_mode = 3
        self.pursuer_hint_refresh = False
        return self.pursuer_search_unit, False

    def _pursuer_has_line_of_sight(self, pursuer_xy: np.ndarray, evader_xy: np.ndarray) -> bool:
        if not np.all(np.isfinite(pursuer_xy)) or not np.all(np.isfinite(evader_xy)):
            return False
        distance = self._distance(pursuer_xy, evader_xy)
        if self.pursuer_line_of_sight_max_distance > 0.0 and distance > self.pursuer_line_of_sight_max_distance:
            return False
        for obstacle in self.pursuer_obstacle_cache:
            if (
                self._segment_to_world_footprint_distance(
                    pursuer_xy,
                    evader_xy,
                    obstacle,
                    precise_margin=self.pursuer_line_of_sight_obstacle_margin,
                )
                < self.pursuer_line_of_sight_obstacle_margin
            ):
                return False
        return True

    def _pursuer_planner_step(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        fallback_unit: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        self.pursuer_planner_active = False
        self.pursuer_planner_replan = False
        self.pursuer_planner_stuck_recovery = False

        if not self.pursuer_planner_enabled or self.pursuer_planner_grid is None:
            return None

        hold_lookahead = pursuer_xy + fallback_unit * min(
            self._distance(pursuer_xy, evader_xy),
            self._pursuer_avoidance_lookahead_distance(step),
        )
        direct_path_is_clear = self._pursuer_path_building_hit(pursuer_xy, evader_xy, evader_xy) is None
        short_path_is_clear = self._pursuer_path_building_hit(pursuer_xy, hold_lookahead, evader_xy) is None

        if direct_path_is_clear or (self.pursuer_line_of_sight and short_path_is_clear):
            self.pursuer_direct_chase_hold_until_step = self.step_count + self.pursuer_direct_chase_hold_steps
            self._clear_pursuer_plan()
            return pursuer_xy + fallback_unit * step, fallback_unit

        if self.step_count <= self.pursuer_direct_chase_hold_until_step and short_path_is_clear:
            self._clear_pursuer_plan()
            return pursuer_xy + fallback_unit * step, fallback_unit

        self.pursuer_planner_active = True
        self._record_pursuer_planner_position(pursuer_xy)

        if self._pursuer_is_planner_stuck():
            recovery = self._pursuer_unstuck_position(pursuer_xy)
            if recovery is not None and float(np.linalg.norm(recovery - pursuer_xy)) > 1e-6:
                self.pursuer_planner_stuck_recovery = True
                self._clear_pursuer_plan()
                unit = self._unit_or_none(recovery - pursuer_xy)
                if unit is None:
                    return pursuer_xy + fallback_unit * step, fallback_unit
                return pursuer_xy + unit * step, unit

        start_cell = self._pursuer_world_to_cell(pursuer_xy)
        goal_cell = self._pursuer_world_to_cell(evader_xy)
        if start_cell is None or goal_cell is None:
            return None

        start_cell = self._nearest_free_pursuer_cell(start_cell)
        goal_cell = self._nearest_free_pursuer_cell(goal_cell)
        if start_cell is None or goal_cell is None:
            return None

        needs_replan = (
            not self.pursuer_planner_path
            or self.pursuer_planner_goal_cell is None
            or self.step_count - self.pursuer_planner_last_replan_step >= self.pursuer_planner_replan_steps
            or self._pursuer_cell_distance(goal_cell, self.pursuer_planner_goal_cell) * self.pursuer_planner_cell_size
            >= self.pursuer_planner_goal_tolerance
        )
        if needs_replan:
            path_cells = self._pursuer_astar(start_cell, goal_cell)
            self.pursuer_planner_replan = True
            self.pursuer_planner_last_replan_step = self.step_count
            self.pursuer_planner_goal_cell = goal_cell
            self.pursuer_planner_path = [self._pursuer_cell_center(cell) for cell in path_cells]

        self._trim_reached_pursuer_waypoints(pursuer_xy)
        self.pursuer_planner_path_length = len(self.pursuer_planner_path)
        if not self.pursuer_planner_path:
            return None

        waypoint = self.pursuer_planner_path[0]
        unit = self._unit_or_none(waypoint - pursuer_xy)
        if unit is None:
            return None
        return pursuer_xy + unit * step, unit

    def _clear_pursuer_plan(self) -> None:
        self.pursuer_planner_path = []
        self.pursuer_planner_path_length = 0
        self.pursuer_planner_goal_cell = None

    def _record_pursuer_planner_position(self, pursuer_xy: np.ndarray) -> None:
        self.pursuer_planner_positions.append(np.asarray(pursuer_xy, dtype=np.float32).copy())
        max_positions = max(self.pursuer_unstuck_window_steps, 2)
        if len(self.pursuer_planner_positions) > max_positions:
            self.pursuer_planner_positions = self.pursuer_planner_positions[-max_positions:]

    def _pursuer_is_planner_stuck(self) -> bool:
        if not self.pursuer_unstuck_enabled:
            return False
        if len(self.pursuer_planner_positions) < self.pursuer_unstuck_window_steps:
            return False
        return self._distance(self.pursuer_planner_positions[0], self.pursuer_planner_positions[-1]) < self.pursuer_unstuck_min_progress

    def _pursuer_unstuck_position(self, pursuer_xy: np.ndarray) -> np.ndarray | None:
        cell = self._pursuer_world_to_cell(pursuer_xy)
        free_cell = self._nearest_free_pursuer_cell(cell) if cell is not None else None
        if free_cell is None:
            return None
        return self._pursuer_cell_center(free_cell)

    def _rebuild_pursuer_planner_grid(self) -> None:
        min_x, max_x, min_y, max_y = self.random_obstacle_bounds
        padding = self.pursuer_planner_padding + self.pursuer_planner_cell_size
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        width = max(1, int(math.ceil((max_x - min_x) / self.pursuer_planner_cell_size)))
        height = max(1, int(math.ceil((max_y - min_y) / self.pursuer_planner_cell_size)))
        self.pursuer_planner_origin = np.array([min_x, min_y], dtype=np.float32)
        self.pursuer_planner_shape = (width, height)
        grid = np.zeros((height, width), dtype=np.bool_)
        self.pursuer_planner_grid = grid

        for obstacle in self.pursuer_obstacle_cache:
            radius = obstacle.radius + self.pursuer_planner_padding
            min_cell = self._pursuer_world_to_cell(obstacle.center - radius)
            max_cell = self._pursuer_world_to_cell(obstacle.center + radius)
            if min_cell is None or max_cell is None:
                continue
            min_cx = max(0, min(min_cell[0], max_cell[0]))
            max_cx = min(width - 1, max(min_cell[0], max_cell[0]))
            min_cy = max(0, min(min_cell[1], max_cell[1]))
            max_cy = min(height - 1, max(min_cell[1], max_cell[1]))
            for cy in range(min_cy, max_cy + 1):
                for cx in range(min_cx, max_cx + 1):
                    center = self._pursuer_cell_center((cx, cy))
                    if self._distance(center, obstacle.center) <= radius:
                        grid[cy, cx] = True

        self._clear_pursuer_plan()

    def _pursuer_world_to_cell(self, xy: np.ndarray) -> tuple[int, int] | None:
        if self.pursuer_planner_grid is None:
            return None
        xy = np.asarray(xy, dtype=np.float32)
        if not np.all(np.isfinite(xy)):
            return None
        local = (xy - self.pursuer_planner_origin) / self.pursuer_planner_cell_size
        cx = int(math.floor(float(local[0])))
        cy = int(math.floor(float(local[1])))
        width, height = self.pursuer_planner_shape
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return None
        return cx, cy

    def _pursuer_cell_center(self, cell: tuple[int, int]) -> np.ndarray:
        return self.pursuer_planner_origin + np.array(
            [
                (cell[0] + 0.5) * self.pursuer_planner_cell_size,
                (cell[1] + 0.5) * self.pursuer_planner_cell_size,
            ],
            dtype=np.float32,
        )

    def _nearest_free_pursuer_cell(self, cell: tuple[int, int] | None) -> tuple[int, int] | None:
        if cell is None or self.pursuer_planner_grid is None:
            return None
        width, height = self.pursuer_planner_shape
        start_x = int(np.clip(cell[0], 0, width - 1))
        start_y = int(np.clip(cell[1], 0, height - 1))
        if not self.pursuer_planner_grid[start_y, start_x]:
            return start_x, start_y
        max_radius = max(width, height)
        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cx = start_x + dx
                    cy = start_y + dy
                    if cx < 0 or cy < 0 or cx >= width or cy >= height:
                        continue
                    if not self.pursuer_planner_grid[cy, cx]:
                        return cx, cy
        return None

    def _pursuer_astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        if self.pursuer_planner_grid is None:
            return []
        if start == goal:
            return [goal]

        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (self._pursuer_cell_distance(start, goal), 0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        expansions = 0

        while open_heap and expansions < self.pursuer_planner_max_expansions:
            _priority, current_cost, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_pursuer_path(came_from, current)
            if current_cost > cost_so_far.get(current, float("inf")) + 1e-6:
                continue
            expansions += 1
            for neighbor, move_cost in self._pursuer_grid_neighbors(current):
                new_cost = current_cost + move_cost
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                priority = new_cost + self._pursuer_cell_distance(neighbor, goal)
                heapq.heappush(open_heap, (priority, new_cost, neighbor))
                came_from[neighbor] = current

        return []

    def _pursuer_grid_neighbors(self, cell: tuple[int, int]) -> tuple[tuple[tuple[int, int], float], ...]:
        if self.pursuer_planner_grid is None:
            return ()
        width, height = self.pursuer_planner_shape
        neighbors: list[tuple[tuple[int, int], float]] = []
        for dx, dy, cost in (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ):
            nx = cell[0] + dx
            ny = cell[1] + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if self.pursuer_planner_grid[ny, nx]:
                continue
            if dx != 0 and dy != 0:
                if self.pursuer_planner_grid[cell[1], nx] or self.pursuer_planner_grid[ny, cell[0]]:
                    continue
            neighbors.append(((nx, ny), cost))
        return tuple(neighbors)

    @staticmethod
    def _pursuer_cell_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
        return math.hypot(float(first[0] - second[0]), float(first[1] - second[1]))

    @staticmethod
    def _reconstruct_pursuer_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _trim_reached_pursuer_waypoints(self, pursuer_xy: np.ndarray) -> None:
        while self.pursuer_planner_path:
            if self._distance(pursuer_xy, self.pursuer_planner_path[0]) > self.pursuer_planner_waypoint_tolerance:
                break
            self.pursuer_planner_path.pop(0)

    def _pursuer_hint_age_seconds(self) -> float:
        if self.pursuer_behavior_mode != "limited_info_patrol":
            return 0.0
        age_steps = max(self.step_count - self.pursuer_last_hint_step, 0)
        return age_steps * (self.timestep * self.action_repeat) / 1000.0

    def _steps_for_seconds(self, seconds: float) -> int:
        step_seconds = max((self.timestep * self.action_repeat) / 1000.0, 1e-6)
        return max(1, int(math.ceil(seconds / step_seconds)))

    @staticmethod
    def _rotate_unit(unit: np.ndarray, angle: float) -> np.ndarray:
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        return np.array(
            [
                float(unit[0]) * cos_angle - float(unit[1]) * sin_angle,
                float(unit[0]) * sin_angle + float(unit[1]) * cos_angle,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared < 1e-9:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
        closest = start + t * segment
        return float(np.linalg.norm(point - closest))

    def _pursuer_collision_avoidance_step(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        direct_unit: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.pursuer_avoidance_active = False
        self.pursuer_avoidance_obstacle_count = 0
        self.pursuer_lidar_avoidance_active = False
        self.pursuer_lidar_front_distance = self.pursuer_lidar_range
        self.pursuer_lidar_left_distance = self.pursuer_lidar_range
        self.pursuer_lidar_right_distance = self.pursuer_lidar_range
        self.pursuer_lidar_danger_count = 0
        direct_next = pursuer_xy + direct_unit * step
        if not self.pursuer_avoid_obstacles or not self.pursuer_obstacle_cache:
            return direct_next, direct_unit

        avoidance_margin = self._pursuer_active_avoidance_margin()
        lookahead_end = pursuer_xy + direct_unit * self._pursuer_avoidance_lookahead_distance(step)
        hit = self._pursuer_path_building_hit(pursuer_xy, lookahead_end, evader_xy, margin=avoidance_margin)
        if hit is None:
            return direct_next, direct_unit
        if self.pursuer_line_of_sight:
            immediate_hit = self._pursuer_path_building_hit(pursuer_xy, direct_next, evader_xy, margin=avoidance_margin)
            if immediate_hit is None:
                return direct_next, direct_unit

        lidar_candidate = self._pursuer_lidar_avoidance_step(pursuer_xy, evader_xy, direct_unit, step, avoidance_margin)
        if lidar_candidate is not None:
            self._start_pursuer_return_to_chase()
            return lidar_candidate

        obstacle = hit
        self.pursuer_avoidance_active = True
        self.pursuer_avoidance_obstacle_count = 1

        away = pursuer_xy - obstacle.center
        away_norm = float(np.linalg.norm(away))
        if away_norm < 1e-6:
            away = -direct_unit
            away_norm = float(np.linalg.norm(away))
        away_unit = away / max(away_norm, 1e-6)
        base_tangent = np.array([-away_unit[1], away_unit[0]], dtype=np.float32)
        side = self._pursuer_avoidance_side_for_obstacle(obstacle, base_tangent, direct_unit)

        same_side_candidates = self._pursuer_avoidance_candidates(direct_unit, base_tangent, away_unit, side)
        candidate = self._first_clear_pursuer_candidate(
            pursuer_xy,
            evader_xy,
            step,
            same_side_candidates,
            margin=avoidance_margin,
        )
        if candidate is not None:
            self._commit_pursuer_avoidance(obstacle, side)
            self._start_pursuer_return_to_chase()
            return candidate

        opposite_side = -side if side != 0 else -1
        opposite_candidates = self._pursuer_avoidance_candidates(direct_unit, base_tangent, away_unit, opposite_side)
        candidate = self._first_clear_pursuer_candidate(
            pursuer_xy,
            evader_xy,
            step,
            opposite_candidates,
            margin=avoidance_margin,
        )
        if candidate is not None:
            self._commit_pursuer_avoidance(obstacle, opposite_side)
            self._start_pursuer_return_to_chase()
            return candidate

        return pursuer_xy, direct_unit

    def _pursuer_simple_collision_avoidance_step(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        direct_unit: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        direct_next = pursuer_xy + direct_unit * step
        if not self.pursuer_avoid_obstacles or not self.pursuer_obstacle_cache:
            return direct_next, direct_unit

        lookahead_end = pursuer_xy + direct_unit * self._pursuer_avoidance_lookahead_distance(step)
        hit = self._pursuer_path_building_hit(
            pursuer_xy,
            lookahead_end,
            evader_xy,
            margin=self._pursuer_active_avoidance_margin(),
        )
        if hit is None:
            return direct_next, direct_unit
        if self.pursuer_line_of_sight:
            immediate_hit = self._pursuer_path_building_hit(
                pursuer_xy,
                direct_next,
                evader_xy,
                margin=self._pursuer_active_avoidance_margin(),
            )
            if immediate_hit is None:
                return direct_next, direct_unit

        away = pursuer_xy - hit.center
        away_unit = self._unit_or_none(away)
        if away_unit is None:
            away_unit = -direct_unit
        tangent = np.array([-away_unit[1], away_unit[0]], dtype=np.float32)
        preferred_side = 1 if float(np.dot(tangent, direct_unit)) >= float(np.dot(-tangent, direct_unit)) else -1
        candidate_units = (
            self._unit_or_none((tangent if preferred_side > 0 else -tangent) + 0.25 * direct_unit),
            self._unit_or_none((-tangent if preferred_side > 0 else tangent) + 0.25 * direct_unit),
            self._unit_or_none(away_unit),
        )
        candidate = self._first_clear_pursuer_candidate(
            pursuer_xy,
            evader_xy,
            step,
            candidate_units,
            margin=self._pursuer_active_avoidance_margin(),
        )
        if candidate is not None:
            self._start_pursuer_return_to_chase()
            return candidate
        return pursuer_xy, direct_unit

    def _pursuer_avoidance_side_for_obstacle(
        self,
        obstacle: WorldFootprint,
        base_tangent: np.ndarray,
        direct_unit: np.ndarray,
    ) -> int:
        if self._pursuer_avoidance_commit_is_active(obstacle):
            return self.pursuer_avoidance_side

        side = 1 if float(np.dot(base_tangent, direct_unit)) >= float(np.dot(-base_tangent, direct_unit)) else -1
        if self.pursuer_avoidance_side != 0 and self.step_count <= self.pursuer_avoidance_commit_until_step:
            return self.pursuer_avoidance_side
        return side

    def _pursuer_avoidance_commit_is_active(self, obstacle: WorldFootprint) -> bool:
        if self.pursuer_avoidance_side == 0 or self.step_count > self.pursuer_avoidance_commit_until_step:
            return False
        if self.pursuer_avoidance_obstacle_center is None:
            return False
        distance = float(np.linalg.norm(obstacle.center - self.pursuer_avoidance_obstacle_center))
        same_obstacle_distance = max(
            self.pursuer_avoidance_same_obstacle_distance,
            0.5 * max(float(obstacle.radius), self.pursuer_avoidance_obstacle_radius),
        )
        return distance <= same_obstacle_distance

    def _commit_pursuer_avoidance(self, obstacle: WorldFootprint, side: int) -> None:
        if side == 0:
            return
        self.pursuer_avoidance_side = int(np.sign(side))
        self.pursuer_avoidance_commit_until_step = self.step_count + self.pursuer_avoidance_commit_steps
        self.pursuer_avoidance_obstacle_center = obstacle.center.copy()
        self.pursuer_avoidance_obstacle_radius = float(obstacle.radius)

    def _start_pursuer_return_to_chase(self) -> None:
        if self.pursuer_return_to_chase_steps <= 0:
            return
        self.pursuer_return_to_chase_until_step = self.step_count + self.pursuer_return_to_chase_steps

    def _pursuer_avoidance_candidates(
        self,
        direct_unit: np.ndarray,
        base_tangent: np.ndarray,
        away_unit: np.ndarray,
        side: int,
    ) -> tuple[np.ndarray | None, ...]:
        tangent = base_tangent if side >= 0 else -base_tangent
        return (
            self._unit_or_none(0.70 * direct_unit + 0.30 * tangent),
            self._unit_or_none(0.50 * direct_unit + 0.50 * tangent),
            self._unit_or_none(0.30 * direct_unit + 0.70 * tangent),
            self._unit_or_none(0.65 * tangent + 0.35 * away_unit),
            self._unit_or_none(tangent),
            self._unit_or_none(away_unit),
        )

    def _first_clear_pursuer_candidate(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        step: float,
        candidate_units: tuple[np.ndarray | None, ...],
        margin: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        for candidate_unit in candidate_units:
            if candidate_unit is None:
                continue
            candidate_xy = pursuer_xy + candidate_unit * step
            candidate_lookahead = pursuer_xy + candidate_unit * self._pursuer_avoidance_lookahead_distance(step)
            hit = self._pursuer_path_building_hit(pursuer_xy, candidate_lookahead, evader_xy, margin=margin)
            if hit is None or self._pursuer_candidate_improves_clearance(pursuer_xy, candidate_xy, hit):
                return candidate_xy, candidate_unit
        return None

    def _pursuer_avoidance_lookahead_distance(self, step: float) -> float:
        lookahead_steps = self.pursuer_avoidance_lookahead_steps
        if self.pursuer_line_of_sight:
            lookahead_steps = min(lookahead_steps, 3.0)
        return max(float(step), float(step) * lookahead_steps)

    def _pursuer_active_avoidance_margin(self) -> float:
        if self.pursuer_line_of_sight:
            return max(0.8, min(self.pursuer_obstacle_margin, self.pursuer_line_of_sight_obstacle_margin))
        return max(self.pursuer_obstacle_margin, 0.2)

    def _pursuer_lidar_avoidance_step(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        direct_unit: np.ndarray,
        step: float,
        margin: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.pursuer_lidar_avoidance_enabled:
            return None

        scan = self._pursuer_virtual_lidar_scan(pursuer_xy, evader_xy, direct_unit)
        self.pursuer_lidar_front_distance = float(scan["front_distance"])
        self.pursuer_lidar_left_distance = float(scan["left_distance"])
        self.pursuer_lidar_right_distance = float(scan["right_distance"])
        self.pursuer_lidar_danger_count = int(scan["danger_count"])

        danger_distance = self.pursuer_lidar_danger_distance
        if scan["front_distance"] >= danger_distance and scan["danger_count"] <= 0:
            return None

        side = self._pursuer_lidar_avoidance_side(scan)
        candidate_units = self._pursuer_lidar_candidates(direct_unit, side)
        best_candidate: tuple[np.ndarray, np.ndarray, float] | None = None
        for candidate_unit in candidate_units:
            if candidate_unit is None:
                continue
            candidate_xy = pursuer_xy + candidate_unit * step
            clearance = self._pursuer_virtual_lidar_distance(pursuer_xy, evader_xy, candidate_unit)
            hit = self._pursuer_path_building_hit(
                pursuer_xy,
                pursuer_xy + candidate_unit * self._pursuer_avoidance_lookahead_distance(step),
                evader_xy,
                margin=margin,
            )
            score = clearance + 2.0 * float(hit is None)
            if best_candidate is None or score > best_candidate[2]:
                best_candidate = (candidate_xy, candidate_unit, score)
            if hit is None and clearance >= danger_distance:
                self.pursuer_avoidance_active = True
                self.pursuer_lidar_avoidance_active = True
                self.pursuer_avoidance_obstacle_count = max(1, int(scan["danger_count"]))
                self.pursuer_avoidance_side = side
                self.pursuer_avoidance_commit_until_step = self.step_count + self.pursuer_avoidance_commit_steps
                return candidate_xy, candidate_unit

        if best_candidate is None:
            return None
        self.pursuer_avoidance_active = True
        self.pursuer_lidar_avoidance_active = True
        self.pursuer_avoidance_obstacle_count = max(1, int(scan["danger_count"]))
        self.pursuer_avoidance_side = side
        self.pursuer_avoidance_commit_until_step = self.step_count + self.pursuer_avoidance_commit_steps
        return best_candidate[0], best_candidate[1]

    def _pursuer_lidar_avoidance_side(self, scan: dict[str, float]) -> int:
        if self.pursuer_avoidance_side != 0 and self.step_count <= self.pursuer_avoidance_commit_until_step:
            return self.pursuer_avoidance_side
        left_risk = float(scan["left_risk"])
        right_risk = float(scan["right_risk"])
        if abs(left_risk - right_risk) < 1e-4:
            return 1
        return -1 if left_risk > right_risk else 1

    def _pursuer_lidar_candidates(self, direct_unit: np.ndarray, side: int) -> tuple[np.ndarray | None, ...]:
        side = 1 if side >= 0 else -1
        tangent = self._rotate_unit(direct_unit, side * math.pi * 0.5)
        opposite_tangent = -tangent
        away = -direct_unit
        return (
            self._unit_or_none(tangent),
            self._unit_or_none(0.75 * tangent + 0.25 * away),
            self._unit_or_none(0.80 * tangent + 0.20 * direct_unit),
            self._unit_or_none(0.55 * tangent + 0.45 * direct_unit),
            self._unit_or_none(away),
            self._unit_or_none(opposite_tangent),
            self._unit_or_none(0.75 * opposite_tangent + 0.25 * away),
        )

    def _pursuer_virtual_lidar_scan(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        heading_unit: np.ndarray,
    ) -> dict[str, float]:
        half_scan = 0.5 * self.pursuer_lidar_scan_angle
        half_front = 0.5 * self.pursuer_lidar_front_angle
        ray_count = max(self.pursuer_lidar_ray_count, 7)
        angles = np.linspace(-half_scan, half_scan, ray_count, dtype=np.float32)

        front_distance = self.pursuer_lidar_range
        left_distance = self.pursuer_lidar_range
        right_distance = self.pursuer_lidar_range
        left_risk = 0.0
        right_risk = 0.0
        danger_count = 0
        for angle in angles:
            angle_float = float(angle)
            ray_unit = self._rotate_unit(heading_unit, angle_float)
            distance = self._pursuer_virtual_lidar_distance(pursuer_xy, evader_xy, ray_unit)
            risk = max(0.0, (self.pursuer_lidar_danger_distance - distance) / self.pursuer_lidar_danger_distance)
            if distance < self.pursuer_lidar_danger_distance:
                danger_count += 1
            if abs(angle_float) <= half_front:
                front_distance = min(front_distance, distance)
            if angle_float > 0.0:
                left_distance = min(left_distance, distance)
                left_risk += risk
            elif angle_float < 0.0:
                right_distance = min(right_distance, distance)
                right_risk += risk

        return {
            "front_distance": float(front_distance),
            "left_distance": float(left_distance),
            "right_distance": float(right_distance),
            "left_risk": float(left_risk),
            "right_risk": float(right_risk),
            "danger_count": float(danger_count),
        }

    def _pursuer_virtual_lidar_distance(
        self,
        pursuer_xy: np.ndarray,
        evader_xy: np.ndarray,
        ray_unit: np.ndarray,
    ) -> float:
        closest = self.pursuer_lidar_range
        expanded_margin = max(self.pursuer_obstacle_margin, 0.2)
        for obstacle in self.pursuer_obstacle_cache:
            if self._point_to_world_footprint_distance(evader_xy, obstacle) <= self.capture_distance:
                continue
            distance = self._ray_to_circle_distance(
                pursuer_xy,
                ray_unit,
                obstacle.center,
                obstacle.radius + expanded_margin,
                self.pursuer_lidar_range,
            )
            closest = min(closest, distance)
        return float(closest)

    @staticmethod
    def _ray_to_circle_distance(
        origin: np.ndarray,
        unit: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_range: float,
    ) -> float:
        offset = center - origin
        projection = float(np.dot(offset, unit))
        perpendicular_squared = float(np.dot(offset, offset) - projection * projection)
        radius_squared = float(radius * radius)
        if perpendicular_squared > radius_squared:
            return max_range
        half_chord = math.sqrt(max(0.0, radius_squared - perpendicular_squared))
        hit_distance = projection - half_chord
        if hit_distance < 0.0:
            hit_distance = projection + half_chord
        if hit_distance < 0.0 or hit_distance > max_range:
            return max_range
        return float(hit_distance)

    def _pursuer_candidate_improves_clearance(
        self,
        pursuer_xy: np.ndarray,
        candidate_xy: np.ndarray,
        obstacle: WorldFootprint,
    ) -> bool:
        current_clearance = self._point_to_world_footprint_distance(pursuer_xy, obstacle)
        candidate_clearance = self._point_to_world_footprint_distance(candidate_xy, obstacle)
        return candidate_clearance > current_clearance + 0.05

    @staticmethod
    def _unit_or_none(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return vector / norm

    def _pursuer_building_hit(
        self,
        next_xy: np.ndarray,
        evader_xy: np.ndarray,
    ) -> WorldFootprint | None:
        closest_hit: WorldFootprint | None = None
        closest_clearance = float("inf")
        for obstacle in self.pursuer_obstacle_cache:
            if (
                self._point_to_world_footprint_distance(
                    evader_xy,
                    obstacle,
                    precise_margin=self.capture_distance,
                )
                <= self.capture_distance
            ):
                continue
            clearance = self._point_to_world_footprint_distance(
                next_xy,
                obstacle,
                precise_margin=max(self.pursuer_obstacle_margin, 0.2),
            )
            if clearance >= max(self.pursuer_obstacle_margin, 0.2):
                continue
            if clearance < closest_clearance:
                closest_clearance = clearance
                closest_hit = obstacle
        return closest_hit

    def _pursuer_path_building_hit(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        evader_xy: np.ndarray,
        margin: float | None = None,
    ) -> WorldFootprint | None:
        closest_hit: WorldFootprint | None = None
        closest_clearance = float("inf")
        active_margin = max(float(self.pursuer_obstacle_margin if margin is None else margin), 0.2)
        for obstacle in self.pursuer_obstacle_cache:
            if self._point_to_world_footprint_distance(evader_xy, obstacle) <= self.capture_distance:
                continue
            clearance = self._segment_to_world_footprint_distance(start_xy, end_xy, obstacle)
            if clearance >= active_margin:
                continue
            if clearance < closest_clearance:
                closest_clearance = clearance
                closest_hit = obstacle
        return closest_hit

    def _observation(self) -> dict[str, np.ndarray]:
        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        delta = pursuer_xy - evader_xy
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        bearing = math.atan2(float(delta[1]), float(delta[0]))

        speed_kmh = self._current_speed_kmh()
        speed_mps = speed_kmh / 3.6
        acceleration_mps2 = self._current_acceleration(speed_mps)
        vision = self._camera_pursuer_observation()
        heading = self._evader_heading()
        lidar_bins = self._lidar_bins()
        front_risk, front_left_risk, front_right_risk = self._front_lidar_risk(
            self.reward_weights.front_obstacle_avoidance_distance
        )
        avoidance = self._avoidance_observation(front_risk, front_left_risk, front_right_risk)
        return self._sanitize_observation({
            "lidar": lidar_bins,
            "vision": np.array(
                [
                    vision["visible"],
                ],
                dtype=np.float32,
            ),
            "pursuer": np.array(
                [
                    np.clip(delta[0] / 120.0, -1.0, 1.0),
                    np.clip(delta[1] / 120.0, -1.0, 1.0),
                    np.clip(distance / 120.0, 0.0, 1.0),
                    math.sin(bearing),
                ],
                dtype=np.float32,
            ),
            "ego": np.array(
                [
                    np.clip(speed_kmh / 70.0, -1.0, 1.0),
                    np.clip(acceleration_mps2 / 8.0, -1.0, 1.0),
                    np.clip(self.current_steering / 0.55, -1.0, 1.0),
                    np.clip(self._yaw_rate() / 2.0, -1.0, 1.0),
                    math.sin(heading),
                    math.cos(heading),
                    1.0 if self._has_touch_contact() else 0.0,
                ],
                dtype=np.float32,
            ),
            "avoidance": avoidance,
        })

    def _lidar_bins(self) -> np.ndarray:
        if self.directional_lidars:
            bins: list[float] = []
            self.directional_lidar_ranges = {}
            for direction in ("front", "left", "back", "right"):
                ranges = self._read_lidar_ranges(self.directional_lidars.get(direction))
                self.directional_lidar_ranges[direction] = ranges
                max_range = self._lidar_max_range(self.directional_lidars.get(direction))
                if ranges.size == 0:
                    bins.extend([1.0, 1.0, 1.0])
                else:
                    chunks = np.array_split(ranges, 3)
                    bins.extend([float(np.min(chunk) / max_range) for chunk in chunks])
            return np.array(bins, dtype=np.float32)

        return np.ones(12, dtype=np.float32)

    def _sanitize_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sanitized: dict[str, np.ndarray] = {}
        for key, value in obs.items():
            low = self.observation_space[key].low
            high = self.observation_space[key].high
            clean = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
            sanitized[key] = np.clip(clean, low, high).astype(np.float32)
        return sanitized

    def _current_speed_kmh(self) -> float:
        if self.driver is None:
            return 0.0
        try:
            speed = float(self.driver.getCurrentSpeed())
        except Exception:
            return 0.0
        if not math.isfinite(speed):
            return 0.0
        return speed

    def _read_lidar_ranges(self, lidar: Any | None) -> np.ndarray:
        if lidar is None:
            return np.array([], dtype=np.float32)
        max_range = self._lidar_max_range(lidar)
        ranges = np.asarray(lidar.getRangeImage(), dtype=np.float32)
        return self._clean_lidar_ranges(ranges, max_range)

    def _current_acceleration(self, speed_mps: float) -> float:
        dt = max((self.timestep * self.action_repeat) / 1000.0, 1e-6)
        return (speed_mps - self.previous_speed_mps) / dt

    def _yaw_rate(self) -> float:
        if self.gyro is None:
            return 0.0
        try:
            values = self.gyro.getValues()
            return float(values[2])
        except Exception:
            return 0.0

    def _avoidance_observation(self, front_risk: float, left_risk: float, right_risk: float) -> np.ndarray:
        progress = self._obstacle_heading_progress(front_risk, left_risk, right_risk)
        danger_delta = float(np.clip(left_risk - right_risk, -1.0, 1.0))
        goal = max(progress["goal"], 1e-6)
        signed_progress = float(np.clip(progress["signed"] / goal, -1.0, 1.0))
        remaining = float(np.clip((goal - max(0.0, progress["signed"])) / goal, 0.0, 1.0))
        if front_risk <= self.reward_weights.obstacle_turn_release_risk:
            remaining = 0.0
        return np.array(
            [
                np.clip(front_risk, 0.0, 1.0),
                danger_delta,
                np.clip(progress["direction"], -1.0, 1.0),
                signed_progress,
                remaining,
            ],
            dtype=np.float32,
        )

    def _obstacle_heading_progress(self, front_risk: float, left_risk: float, right_risk: float) -> dict[str, float]:
        active = front_risk > self.reward_weights.obstacle_action_risk_threshold
        if not active:
            if front_risk <= self.reward_weights.obstacle_turn_release_risk:
                self.obstacle_avoidance_start_heading = None
            return {"direction": 0.0, "signed": 0.0, "raw": 0.0, "goal": self._required_obstacle_heading_change(front_risk)}

        if self.obstacle_avoidance_start_heading is None:
            self.obstacle_avoidance_start_heading = self._evader_heading()

        direction = self._obstacle_turn_direction_hint(left_risk, right_risk)
        raw_delta = self._wrap_angle(self._evader_heading() - self.obstacle_avoidance_start_heading)
        signed_delta = direction * raw_delta if direction != 0 else 0.0
        return {
            "direction": float(direction),
            "signed": float(signed_delta),
            "raw": float(raw_delta),
            "goal": self._required_obstacle_heading_change(front_risk),
        }

    def _obstacle_turn_direction_hint(self, left_risk: float, right_risk: float) -> int:
        committed = int(getattr(self, "obstacle_turn_direction", 0))
        if committed != 0:
            return committed

        danger_delta = left_risk - right_risk
        if abs(danger_delta) > self.reward_weights.obstacle_turn_direction_signal_threshold:
            return 1 if danger_delta > 0.0 else -1

        current_sign = self._sign(float(self.current_steering))
        if current_sign != 0:
            return current_sign

        previous_sign = self._sign(float(self.previous_action[0]))
        if previous_sign != 0:
            return previous_sign
        return 0

    def _required_obstacle_heading_change(self, front_risk: float) -> float:
        return float(np.clip(0.25 + 0.65 * front_risk, 0.25, 0.90))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _enable_camera(self, camera: Any | None, sensor_timestep: int) -> None:
        if camera is None:
            return
        try:
            camera.enable(sensor_timestep)
        except Exception:
            return
        for method_name in ("recognitionEnable", "enableRecognition", "recognition_enable"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                method(sensor_timestep)
                self.recognition_enabled_cameras.add(id(camera))
                return
            except Exception:
                continue
        recognition = self._camera_recognition(camera)
        if recognition is not None:
            try:
                recognition.enable(sensor_timestep)
                self.recognition_enabled_cameras.add(id(camera))
            except Exception:
                pass

    def _camera_recognition(self, camera: Any) -> Any | None:
        for method_name in ("getRecognition", "get_recognition"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                return method()
            except Exception:
                continue
        return None

    def _camera_pursuer_observation(self) -> dict[str, float]:
        front = self._camera_detection(self.front_camera)
        back = self._camera_detection(self.back_camera)
        left = self._camera_detection(self.left_camera)
        right = self._camera_detection(self.right_camera)
        visual_size = max(front["size"], back["size"], left["size"], right["size"])
        return {
            "visible": 1.0 if front["visible"] or back["visible"] or left["visible"] or right["visible"] else 0.0,
            "front_visible": 1.0 if front["visible"] else 0.0,
            "front_x": front["x"],
            "front_bearing": front["bearing"],
            "back_visible": 1.0 if back["visible"] else 0.0,
            "back_x": back["x"],
            "back_bearing": back["bearing"],
            "left_visible": 1.0 if left["visible"] else 0.0,
            "left_x": left["x"],
            "left_bearing": left["bearing"],
            "left_size": left["size"],
            "right_visible": 1.0 if right["visible"] else 0.0,
            "right_x": right["x"],
            "right_bearing": right["bearing"],
            "right_size": right["size"],
            "visual_size": visual_size,
        }

    def _camera_detection(self, camera: Any | None) -> dict[str, float | bool]:
        if camera is None:
            return {"visible": False, "x": 0.0, "bearing": 0.0, "size": 0.0}

        objects = self._recognized_objects(camera)
        best_size = 0.0
        best_x = 0.0
        best_bearing = 0.0
        for obj in objects:
            if not self._is_pursuer_recognition_object(obj):
                continue
            width, height = self._recognition_size(obj)
            size = self._normalized_recognition_area(camera, width, height)
            if size <= best_size:
                continue
            best_size = size
            best_x = self._normalized_recognition_x(camera, obj)
            best_bearing = self._normalized_recognition_bearing(camera, best_x)

        return {"visible": best_size > 0.0, "x": best_x, "bearing": best_bearing, "size": best_size}

    def _recognized_objects(self, camera: Any) -> list[Any]:
        if id(camera) not in self.recognition_enabled_cameras:
            return []

        recognition = self._camera_recognition(camera)
        if recognition is not None:
            for method_name in ("getObjects", "get_objects"):
                method = getattr(recognition, method_name, None)
                if method is None:
                    continue
                try:
                    return list(method())
                except Exception:
                    continue

        for method_name in ("getRecognitionObjects", "get_recognition_objects"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                return list(method())
            except Exception:
                continue
        return []

    def _is_pursuer_recognition_object(self, obj: Any) -> bool:
        candidates = []
        for method_name in ("getModel", "get_model"):
            method = getattr(obj, method_name, None)
            if method is not None:
                try:
                    candidates.append(str(method()))
                except Exception:
                    pass
        for attr_name in ("model", "name"):
            try:
                candidates.append(str(getattr(obj, attr_name)))
            except Exception:
                pass

        text = " ".join(candidates)
        if any(token in text for token in self.pursuer_recognition_tokens):
            return True

        colors = self._recognition_colors(obj)
        return bool(colors and colors[0] > 0.8 and colors[1] < 0.2 and colors[2] < 0.2)

    def _recognition_colors(self, obj: Any) -> tuple[float, float, float] | None:
        for method_name in ("getColors", "get_colors"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                colors = method()
                return float(colors[0]), float(colors[1]), float(colors[2])
            except Exception:
                continue
        return None

    def _recognition_size(self, obj: Any) -> tuple[float, float]:
        for method_name in ("getSizeOnImage", "get_size_on_image"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                size = method()
                return float(size[0]), float(size[1])
            except Exception:
                continue
        return 0.0, 0.0

    def _normalized_recognition_area(self, camera: Any, width: float, height: float) -> float:
        image_width = max(float(self._device_dimension(camera, "width")), 1.0)
        image_height = max(float(self._device_dimension(camera, "height")), 1.0)
        return float(np.clip((width * height) / (image_width * image_height), 0.0, 1.0))

    def _normalized_recognition_x(self, camera: Any, obj: Any) -> float:
        for method_name in ("getPositionOnImage", "get_position_on_image"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                position = method()
                image_width = max(float(self._device_dimension(camera, "width")), 1.0)
                return float(np.clip((2.0 * float(position[0]) / image_width) - 1.0, -1.0, 1.0))
            except Exception:
                continue
        return 0.0

    def _normalized_recognition_bearing(self, camera: Any, normalized_x: float) -> float:
        half_fov = self._camera_fov(camera) * 0.5
        bearing_rad = float(normalized_x) * half_fov
        return float(np.clip(bearing_rad / math.pi, -1.0, 1.0))

    def _camera_fov(self, camera: Any) -> float:
        for method_name in ("getFov", "get_fov"):
            method = getattr(camera, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return 1.0

    def _device_dimension(self, device: Any, dimension: str) -> int:
        method_names = {
            "width": ("getWidth", "get_width"),
            "height": ("getHeight", "get_height"),
        }[dimension]
        for method_name in method_names:
            method = getattr(device, method_name, None)
            if method is None:
                continue
            try:
                return int(method())
            except Exception:
                continue
        return 1

    def _evader_tilt_angle(self) -> float:
        if self.evader_node is None:
            return 0.0
        try:
            orientation = self.evader_node.getOrientation()
            local_up_z = float(orientation[8])
            return float(math.acos(np.clip(local_up_z, -1.0, 1.0)))
        except Exception:
            return 0.0

    def _apply_obstacle_safety(self, steering: float, drive: float) -> tuple[float, float, bool]:
        if not self.obstacle_safety_enabled:
            return steering, drive, False

        front_ranges = self.directional_lidar_ranges.get("front", np.array([], dtype=np.float32))
        front_ranges = self._filter_lidar_ranges(front_ranges)
        if front_ranges.size == 0:
            return steering, drive, False

        ray_offsets = np.linspace(-1.0, 1.0, front_ranges.size, dtype=np.float32)
        corridor_width = max(self.obstacle_safety_corridor_width, 0.05)
        corridor_mask = np.abs(ray_offsets) <= corridor_width
        corridor_ranges = front_ranges[corridor_mask] if np.any(corridor_mask) else front_ranges
        corridor_min = float(np.min(corridor_ranges))

        slow_distance = max(self.obstacle_safety_slow_distance, 1.0)
        brake_distance = min(max(self.obstacle_safety_brake_distance, 0.1), slow_distance)
        center_weights = np.exp(-(ray_offsets**2) / (2.0 * corridor_width**2))
        ray_risk = np.clip((slow_distance - front_ranges) / slow_distance, 0.0, 1.0)
        front_risk = float(np.max(center_weights * ray_risk))

        sector_distances = self._lidar_sector_distances()
        left_close = sector_distances["left"] < 2.8
        right_close = sector_distances["right"] < 2.8
        active = corridor_min < slow_distance or front_risk > 0.18

        if not active:
            if left_close and steering < 0.0:
                return 0.0, min(drive, 0.2), True
            if right_close and steering > 0.0:
                return 0.0, min(drive, 0.2), True
            return steering, drive, False

        midpoint = front_ranges.size // 2
        left_risk = float(np.mean(ray_risk[:midpoint])) if midpoint > 0 else 0.0
        right_risk = float(np.mean(ray_risk[midpoint:])) if midpoint < front_ranges.size else 0.0
        risk_delta = left_risk - right_risk
        if abs(risk_delta) > 0.03:
            desired_sign = 1.0 if risk_delta > 0.0 else -1.0
        else:
            desired_sign = 1.0 if sector_distances["right"] >= sector_distances["left"] else -1.0

        if desired_sign > 0.0 and right_close and not left_close:
            desired_sign = -1.0
        elif desired_sign < 0.0 and left_close and not right_close:
            desired_sign = 1.0

        denominator = max(slow_distance - brake_distance, 1e-6)
        urgency = float(np.clip((slow_distance - corridor_min) / denominator, 0.0, 1.0))
        required_steering = min(0.5, max(self.obstacle_safety_min_steering, 0.25 + 0.25 * urgency))
        adjusted_steering = steering
        if steering * desired_sign <= 0.0 or abs(steering) < required_steering:
            adjusted_steering = self._quantized_steering(desired_sign * required_steering)

        adjusted_drive = drive
        if corridor_min <= brake_distance:
            adjusted_drive = min(adjusted_drive, -0.25)
        elif corridor_min < slow_distance:
            adjusted_drive = min(adjusted_drive, 0.2)

        return adjusted_steering, adjusted_drive, True

    def _stabilize_evader_steering(self, target_steering: float) -> float:
        target_steering = float(np.clip(target_steering, -0.55, 0.55))
        committed_target = target_steering

        front_risk, _, _ = self._front_lidar_risk(self.reward_weights.front_obstacle_avoidance_distance)
        if front_risk <= self.evader_obstacle_steering_release_risk:
            self.evader_steering_commit_direction = 0
            self.evader_steering_commit_until_step = 0
        elif self.evader_obstacle_steering_commit_steps > 0:
            target_sign = self._sign(target_steering)
            target_is_avoidance_turn = (
                target_sign != 0
                and abs(target_steering) >= self.evader_obstacle_steering_min_abs - 1e-6
                and front_risk >= self.evader_obstacle_steering_commit_risk
            )
            commit_active = (
                self.evader_steering_commit_direction != 0
                and self.step_count <= self.evader_steering_commit_until_step
            )

            if target_is_avoidance_turn and not commit_active:
                self.evader_steering_commit_direction = target_sign
                self.evader_steering_commit_until_step = self.step_count + self.evader_obstacle_steering_commit_steps
                commit_active = True
            elif target_is_avoidance_turn and target_sign == self.evader_steering_commit_direction:
                self.evader_steering_commit_until_step = self.step_count + self.evader_obstacle_steering_commit_steps
                commit_active = True

            if commit_active and self.evader_steering_commit_direction != 0:
                if target_sign != self.evader_steering_commit_direction:
                    committed_target = self.evader_steering_commit_direction * max(
                        abs(target_steering),
                        self.evader_obstacle_steering_min_abs,
                    )

        return float(np.clip(committed_target, -0.55, 0.55))

    def _drive_command_from_action(self, drive: float) -> tuple[float, float, float]:
        deadzone = 0.08
        speed_limit_kmh = self._evader_speed_limit_mps() * 3.6
        if drive > deadzone:
            drive_fraction = float((drive - deadzone) / (1.0 - deadzone))
            throttle = 0.18 + 0.82 * drive_fraction
            target_speed = min(8.0 + 52.0 * drive_fraction, speed_limit_kmh)
            if abs(self._current_speed_kmh()) > speed_limit_kmh:
                return 0.0, 0.6, speed_limit_kmh
            return throttle, 0.0, target_speed
        if drive < -deadzone:
            brake = float((-drive - deadzone) / (1.0 - deadzone))
            return 0.0, brake, 0.0
        return 0.0, 0.0, 0.0

    def _evader_speed_limit_mps(self) -> float:
        return max(0.0, self.pursuer_speed_mps + self.evader_speed_margin_mps)

    def _decode_action(self, action: np.ndarray) -> tuple[float, float, float, float, int, int]:
        action_array = np.asarray(action).reshape(-1)
        if self.discrete_actions:
            steering_index = int(np.clip(round(float(action_array[0])), 0, self.steering_targets.size - 1))
            drive_index = int(np.clip(round(float(action_array[1])), 0, self.drive_targets.size - 1))
            steering = float(self.steering_targets[steering_index])
            drive = float(self.drive_targets[drive_index])
            return steering, drive, steering, drive, steering_index, drive_index

        policy_steering = float(np.clip(action_array[0], -0.55, 0.55))
        policy_drive = float(np.clip(action_array[1], -1.0, 1.0))
        steering = self._quantized_steering(policy_steering)
        drive = policy_drive
        steering_index = int(np.argmin(np.abs(self.steering_targets - steering)))
        drive_index = int(np.argmin(np.abs(self.drive_targets - drive)))
        return policy_steering, policy_drive, steering, drive, steering_index, drive_index

    def _quantized_steering(self, steering_action: float) -> float:
        steering_action = float(np.clip(steering_action, -0.55, 0.55))
        target_index = int(np.argmin(np.abs(self.steering_targets - steering_action)))
        return float(self.steering_targets[target_index])

    def _update_steering_streak(self, steering: float) -> None:
        if abs(steering) < 1e-6:
            self.steering_direction_streak = 0
            self.previous_steering_direction = 0
            return

        direction = -1 if steering < 0.0 else 1
        if direction == self.previous_steering_direction:
            self.steering_direction_streak += 1
        else:
            self.steering_direction_streak = 1
            self.previous_steering_direction = direction

    def _lidar_max_range(self, lidar: Any | None = None) -> float:
        lidar = lidar or next(iter(self.directional_lidars.values()), None)
        if lidar is None:
            return 1.0
        return max(float(lidar.getMaxRange()), 1e-6)

    def _lidar_sector_distances(self) -> dict[str, float]:
        if self.directional_lidar_ranges:
            return {
                direction: self._sector_min_distance(self.directional_lidar_ranges.get(direction, np.array([], dtype=np.float32)))
                for direction in ("front", "left", "right", "back")
            }

        max_range = self._lidar_max_range()
        return {"front": max_range, "left": max_range, "right": max_range, "back": max_range}

    def _sector_min_distance(self, ranges: np.ndarray) -> float:
        filtered = self._filter_lidar_ranges(ranges)
        if filtered.size == 0:
            return self._lidar_max_range()
        return float(np.min(filtered))

    def _filter_lidar_ranges(self, ranges: np.ndarray) -> np.ndarray:
        if ranges.size == 0:
            return ranges
        return self._clean_lidar_ranges(ranges)

    def _clean_lidar_ranges(self, ranges: np.ndarray, max_range: float | None = None) -> np.ndarray:
        if ranges.size == 0:
            return ranges
        lidar_max_range = max_range if max_range is not None else self._lidar_max_range()
        clean = np.asarray(ranges, dtype=np.float32).copy()
        clean = np.nan_to_num(clean, nan=lidar_max_range, posinf=lidar_max_range, neginf=0.0)
        clean = np.clip(clean, 0.0, lidar_max_range)
        clean[clean <= self.self_lidar_ignore_distance] = lidar_max_range
        return clean

    def _moved_distance(self, evader_xy: np.ndarray) -> float:
        if self.previous_position is None:
            return 0.0
        if not np.all(np.isfinite(evader_xy)) or not np.all(np.isfinite(self.previous_position)):
            return 0.0
        return float(np.linalg.norm(evader_xy - self.previous_position))

    def _exploration_reward(self) -> float:
        cell = self._exploration_cell(self._evader_xy())
        if cell in self.visited_exploration_cells:
            return self.reward_weights.exploration_revisit_penalty
        self.visited_exploration_cells.add(cell)
        return self.reward_weights.exploration_new_cell_reward

    def _exploration_cell(self, xy: np.ndarray) -> tuple[int, int]:
        if not np.all(np.isfinite(xy)):
            xy = self._evader_translation_xy()
        return (
            int(math.floor(float(xy[0]) / self.exploration_cell_size)),
            int(math.floor(float(xy[1]) / self.exploration_cell_size)),
        )

    def _has_collision(self, min_lidar: float, obstacle_distance: float | None = None) -> bool:
        if obstacle_distance is not None and obstacle_distance <= self.obstacle_collision_distance:
            return True
        if min_lidar <= 0.015:
            return True
        if self._distance(self._evader_xy(), self._pursuer_xy()) <= self.vehicle_touch_distance:
            return True
        if self.step_count <= self.reset_touch_grace_steps:
            return False
        return self._raw_touch_sensor_contact() and self._touch_sensor_contact_is_plausible(min_lidar)

    def _has_touch_contact(self) -> bool:
        if self._distance(self._evader_xy(), self._pursuer_xy()) <= self.vehicle_touch_distance:
            return True
        if self.step_count <= self.reset_touch_grace_steps:
            return False
        return self._raw_touch_sensor_contact() and self._touch_sensor_contact_is_plausible()

    def _raw_touch_sensor_contact(self) -> bool:
        if self.touch_sensor is None:
            return False
        try:
            return bool(self.touch_sensor.getValue() > 0.0)
        except Exception:
            return False

    def _touch_sensor_contact_is_plausible(self, min_lidar: float | None = None) -> bool:
        if min_lidar is not None:
            closest_obstacle = min_lidar * self._lidar_max_range()
        else:
            sector_distances = self._lidar_sector_distances()
            closest_obstacle = min(sector_distances.values())
        return closest_obstacle <= self.touch_collision_lidar_distance

    def _evader_xy(self) -> np.ndarray:
        supervisor_xy = self._evader_translation_xy()
        if self.evader_translation_field is not None and np.all(np.isfinite(supervisor_xy)):
            return supervisor_xy
        if self.gps is not None:
            values = self.gps.getValues()
            xy = np.array([values[0], values[1]], dtype=np.float32)
            if np.all(np.isfinite(xy)):
                return xy
        return supervisor_xy

    def _evader_translation_xy(self) -> np.ndarray:
        if self.evader_translation_field is not None:
            values = self.evader_translation_field.getSFVec3f()
            return np.array([values[0], values[1]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _evader_heading(self) -> float:
        if self.evader_rotation_field is None:
            return 0.0
        try:
            rotation = self.evader_rotation_field.getSFRotation()
        except Exception:
            return 0.0
        axis_z = float(rotation[2])
        angle = float(rotation[3])
        return angle if axis_z >= 0.0 else -angle

    def _pursuer_xy(self) -> np.ndarray:
        if self.pursuer_agents:
            evader_xy = self._evader_translation_xy()
            closest_xy: np.ndarray | None = None
            closest_distance = float("inf")
            for agent in self.pursuer_agents:
                xy = self._pursuer_agent_xy(agent)
                distance = self._distance(xy, evader_xy)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_xy = xy
            if closest_xy is not None:
                return closest_xy

        if self.pursuer_translation_field is not None:
            values = self.pursuer_translation_field.getSFVec3f()
            return np.array([values[0], values[1]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _pursuer_agent_xy(self, agent: dict[str, Any]) -> np.ndarray:
        translation_field = agent.get("translation_field")
        if translation_field is not None:
            try:
                values = translation_field.getSFVec3f()
                return np.array([values[0], values[1]], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1e6
        return float(np.linalg.norm(a - b))


if "Evader-v0" not in registry:
    register(
        id="Evader-v0",
        entry_point="controllers.evader_env:EvaderEnv",
    )

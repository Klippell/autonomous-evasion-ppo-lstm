import math
import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from controllers.evader_env.debug_display import DebugDisplayMixin
from controllers.evader_env.reward import RewardMixin, reward_weights_from_mapping
from controllers.evader_env.webots_runtime import DEFAULT_SPAWN_POSES, SpawnPose, configure_webots_paths


class EvaderEnv(RewardMixin, DebugDisplayMixin, gym.Env):
    """Webots Gym environment for training an evader against a simple pursuer."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 2500,
        capture_distance: float = 6.0,
        vehicle_touch_distance: float = 5.5,
        pursuer_speed_mps: float = 6.0,
        robot_name: str | None = None,
        still_distance_threshold: float = 0.1,
        self_lidar_ignore_distance: float = 0.06,
        action_repeat: int = 4,
        pursuer_start_delay_steps: int = 120,
        show_reward_display: bool = True,
        reward_display_interval: int = 1,
        show_car_display: bool = False,
        reset_vehicle_physics: bool = True,
        exploration_cell_size: float = 8.0,
        sensor_timestep: int | None = None,
        enable_camera_recognition: bool = True,
        randomize_obstacles: bool = False,
        center_spawn_when_random_obstacles: bool = True,
        random_obstacle_def_names: tuple[str, ...] | list[str] = (),
        random_obstacle_bounds: tuple[float, float, float, float] | list[float] = (-170.0, 170.0, -190.0, 260.0),
        random_obstacle_exclusion_center: tuple[float, float] | list[float] = (0.0, 0.0),
        random_obstacle_exclusion_radius: float = 50.0,
        random_obstacle_min_spacing: float = 12.0,
        reward_weights: dict[str, object] | None = None,
        front_camera_names: tuple[str, ...] | list[str] = ("front camera", "front Camera", "camera"),
        back_camera_names: tuple[str, ...] | list[str] = ("back camera", "rear camera"),
        pursuer_recognition_tokens: tuple[str, ...] | list[str] = ("pursuer", "Pursuer"),
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.capture_distance = capture_distance
        self.vehicle_touch_distance = vehicle_touch_distance
        self.pursuer_speed_mps = pursuer_speed_mps
        self.robot_name = robot_name or os.environ.get("WEBOTS_ROBOT_NAME") or "evader"
        self.still_distance_threshold = still_distance_threshold
        self.self_lidar_ignore_distance = self_lidar_ignore_distance
        self.action_repeat = action_repeat
        self.pursuer_start_delay_steps = pursuer_start_delay_steps
        self.show_reward_display = show_reward_display
        self.reward_display_interval = reward_display_interval
        self.show_car_display = show_car_display
        self.reset_vehicle_physics = reset_vehicle_physics
        self.exploration_cell_size = max(float(exploration_cell_size), 1.0)
        self.sensor_timestep = sensor_timestep
        self.enable_camera_recognition = enable_camera_recognition
        self.randomize_obstacles = randomize_obstacles
        self.center_spawn_when_random_obstacles = center_spawn_when_random_obstacles
        self.random_obstacle_def_names = tuple(random_obstacle_def_names)
        self.random_obstacle_bounds = tuple(float(value) for value in random_obstacle_bounds)
        self.random_obstacle_exclusion_center = np.array(random_obstacle_exclusion_center, dtype=np.float32)
        self.random_obstacle_exclusion_radius = float(random_obstacle_exclusion_radius)
        self.random_obstacle_min_spacing = float(random_obstacle_min_spacing)
        self.reward_weights = reward_weights_from_mapping(reward_weights)
        self.front_camera_names = tuple(front_camera_names)
        self.back_camera_names = tuple(back_camera_names)
        self.pursuer_recognition_tokens = tuple(pursuer_recognition_tokens)

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
                "ego": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
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
        self.evader_node: Any = None
        self.pursuer_node: Any = None
        self.evader_translation_field: Any = None
        self.evader_rotation_field: Any = None
        self.pursuer_translation_field: Any = None
        self.pursuer_rotation_field: Any = None
        self.random_obstacles: list[tuple[Any, Any, Any | None, float]] = []

        self.timestep = 32
        self.step_count = 0
        self.previous_distance = 0.0
        self.previous_sector_distances = {"front": 25.0, "left": 25.0, "right": 25.0, "back": 25.0}
        self.previous_position: np.ndarray | None = None
        self.previous_speed_mps = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.current_steering = 0.0
        self.last_reward_log_step = 0
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
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_speed_mps = 0.0
        self.visited_exploration_cells.clear()

        pose = self._spawn_pose_for_episode()
        if self.evader_translation_field is not None:
            self._set_vehicle_pose(
                self.evader_node,
                self.evader_translation_field,
                self.evader_rotation_field,
                pose.evader_xy,
                pose.heading,
                z=0.45,
            )
        if self.pursuer_translation_field is not None:
            self._set_vehicle_pose(
                self.pursuer_node,
                self.pursuer_translation_field,
                self.pursuer_rotation_field,
                pose.pursuer_xy,
                pose.heading,
                z=0.55,
            )
        if self.randomize_obstacles:
            self._randomize_obstacle_poses()

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(0.0)
        self.driver.setThrottle(0.0)
        self.driver.setBrakeIntensity(1.0)
        self.driver.setSteeringAngle(0.0)
        self._step_simulation(5)

        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        self.previous_position = evader_xy
        self.previous_distance = self._distance(evader_xy, pursuer_xy)
        self.visited_exploration_cells.add(self._exploration_cell(evader_xy))
        self._observation()
        self.previous_sector_distances = self._lidar_sector_distances()
        initial_vision = self._camera_pursuer_observation()
        self.previous_pursuer_visible = bool(initial_vision["visible"])
        self.previous_pursuer_visual_size = float(initial_vision["visual_size"])
        return self._observation(), {"distance_to_pursuer": self.previous_distance}

    def step(self, action: np.ndarray):
        self._ensure_webots()
        self.step_count += 1

        steering = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        drive = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))
        clipped_action = np.array([steering, drive], dtype=np.float32)
        self.current_steering = steering
        throttle, brake, target_speed = self._drive_command_from_action(drive)

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(target_speed)
        self.driver.setSteeringAngle(steering)
        self.driver.setThrottle(throttle)
        self.driver.setBrakeIntensity(brake)

        if self.step_count > self.pursuer_start_delay_steps:
            self._move_pursuer()
        self._step_simulation(self.action_repeat)

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
        reward, reward_parts = self._reward(distance, sector_distances, clipped_action, speed_kmh, moved_distance)
        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        if distance <= self.capture_distance:
            reward -= 150.0
            terminated = True
        if self._has_collision(min_lidar):
            reward -= 100.0
            terminated = True

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
            "captured": distance <= self.capture_distance,
            "touch_contact": self._has_touch_contact(),
            "moved_distance": moved_distance,
            "target_speed": target_speed,
            "throttle": throttle,
            "brake": brake,
            "speed_kmh": speed_kmh,
        }
        if self.show_reward_display and self.step_count % self.reward_display_interval == 0:
            self._draw_reward_label(reward, info)
            self._draw_supervisor_minimap(info)
            if self.show_car_display:
                self._draw_reward_display(reward, info)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.driver is not None:
            self.driver.setThrottle(0.0)
            self.driver.setBrakeIntensity(1.0)

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

        for lidar in self.directional_lidars.values():
            lidar.enable(sensor_timestep)
        if self.enable_camera_recognition:
            for camera in (self.front_camera, self.back_camera):
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
        self._collect_random_obstacles()

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

    def _step_simulation(self, substeps: int) -> None:
        for _ in range(substeps):
            if self.driver.step() == -1:
                break

    def _spawn_pose_for_episode(self) -> SpawnPose:
        if self.randomize_obstacles and self.center_spawn_when_random_obstacles:
            return SpawnPose((0.0, 0.0), (-55.0, 0.0), 0.0)
        return self.spawn_poses[0]

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

    def _collect_random_obstacles(self) -> None:
        self.random_obstacles = []
        for def_name in self.random_obstacle_def_names:
            node = self._node_by_def(def_name)
            if node is None:
                print(f"Random obstacle DEF '{def_name}' was not found.")
                continue
            translation_field = node.getField("translation")
            if translation_field is None:
                print(f"Random obstacle DEF '{def_name}' has no translation field.")
                continue
            rotation_field = node.getField("rotation")
            try:
                z = float(translation_field.getSFVec3f()[2])
            except Exception:
                z = 0.0
            self.random_obstacles.append((node, translation_field, rotation_field, z))

    def _randomize_obstacle_poses(self) -> None:
        if not self.random_obstacles:
            return

        placed: list[np.ndarray] = []
        for node, translation_field, rotation_field, z in self.random_obstacles:
            xy = self._sample_random_obstacle_xy(placed)
            placed.append(xy)
            translation_field.setSFVec3f([float(xy[0]), float(xy[1]), z])
            if rotation_field is not None:
                yaw = float(self.np_random.uniform(-math.pi, math.pi))
                rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])
            try:
                node.resetPhysics()
            except Exception:
                pass

    def _sample_random_obstacle_xy(self, placed: list[np.ndarray]) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.random_obstacle_bounds
        for _ in range(500):
            xy = np.array(
                [
                    self.np_random.uniform(min_x, max_x),
                    self.np_random.uniform(min_y, max_y),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(xy - self.random_obstacle_exclusion_center) < self.random_obstacle_exclusion_radius:
                continue
            if any(np.linalg.norm(xy - other) < self.random_obstacle_min_spacing for other in placed):
                continue
            return xy

        return np.array([min_x, min_y], dtype=np.float32)

    def _move_pursuer(self) -> None:
        if self.pursuer_translation_field is None or self.pursuer_rotation_field is None:
            return
        pursuer_xy = self._pursuer_xy()
        evader_xy = self._evader_xy()
        direction = evader_xy - pursuer_xy
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            return

        dt = (self.timestep * self.action_repeat) / 1000.0
        step = min(distance, self.pursuer_speed_mps * dt)
        unit = direction / distance
        next_xy = pursuer_xy + unit * step
        heading = math.atan2(float(unit[1]), float(unit[0]))
        current = self.pursuer_translation_field.getSFVec3f()
        self.pursuer_translation_field.setSFVec3f([float(next_xy[0]), float(next_xy[1]), current[2]])
        self.pursuer_rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])

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
        return self._sanitize_observation({
            "lidar": self._lidar_bins(),
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
                    1.0 if self._has_touch_contact() else 0.0,
                ],
                dtype=np.float32,
            ),
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
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        return np.clip(ranges, 0.0, max_range)

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
        visual_size = max(front["size"], back["size"])
        return {
            "visible": 1.0 if front["visible"] or back["visible"] else 0.0,
            "front_visible": 1.0 if front["visible"] else 0.0,
            "front_x": front["x"],
            "front_bearing": front["bearing"],
            "back_visible": 1.0 if back["visible"] else 0.0,
            "back_x": back["x"],
            "back_bearing": back["bearing"],
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

    @staticmethod
    def _drive_command_from_action(drive: float) -> tuple[float, float, float]:
        deadzone = 0.08
        if drive > deadzone:
            drive_fraction = float((drive - deadzone) / (1.0 - deadzone))
            throttle = 0.18 + 0.82 * drive_fraction
            target_speed = 8.0 + 52.0 * drive_fraction
            return throttle, 0.0, target_speed
        if drive < -deadzone:
            brake = float((-drive - deadzone) / (1.0 - deadzone))
            return 0.0, brake, 0.0
        return 0.0, 0.0, 0.0

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
        max_range = self._lidar_max_range()
        return ranges[(ranges > self.self_lidar_ignore_distance) & (ranges < max_range)]

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

    def _has_collision(self, min_lidar: float) -> bool:
        return self._has_touch_contact() or min_lidar <= 0.015

    def _has_touch_contact(self) -> bool:
        if self._distance(self._evader_xy(), self._pursuer_xy()) <= self.vehicle_touch_distance:
            return True
        if self.touch_sensor is None:
            return False
        try:
            return bool(self.touch_sensor.getValue() > 0.0)
        except Exception:
            return False

    def _evader_xy(self) -> np.ndarray:
        if self.gps is not None:
            values = self.gps.getValues()
            xy = np.array([values[0], values[1]], dtype=np.float32)
            if np.all(np.isfinite(xy)):
                return xy
        return self._evader_translation_xy()

    def _evader_translation_xy(self) -> np.ndarray:
        if self.evader_translation_field is not None:
            values = self.evader_translation_field.getSFVec3f()
            return np.array([values[0], values[1]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _pursuer_xy(self) -> np.ndarray:
        if self.pursuer_translation_field is not None:
            values = self.pursuer_translation_field.getSFVec3f()
            return np.array([values[0], values[1]], dtype=np.float32)
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

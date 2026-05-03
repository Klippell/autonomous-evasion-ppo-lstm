import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry


WEBOTS_HOME = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")


def _configure_webots_paths() -> None:
    os.environ.setdefault("WEBOTS_HOME", WEBOTS_HOME)
    for path in (
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin"),
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "cpp"),
        os.path.join(WEBOTS_HOME, "lib", "controller"),
        os.path.join(WEBOTS_HOME, "projects", "vehicles", "lib"),
    ):
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except (AttributeError, FileNotFoundError, OSError):
                pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    controller_python = os.path.join(WEBOTS_HOME, "lib", "controller", "python")
    if os.path.exists(controller_python) and controller_python not in sys.path:
        sys.path.append(controller_python)


@dataclass(frozen=True)
class SpawnPose:
    evader_xy: tuple[float, float]
    pursuer_xy: tuple[float, float]
    heading: float


class EvaderEnv(gym.Env):
    """Webots Gym environment for training an evader against a simple pursuer."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 2500,
        capture_distance: float = 6.0,
        safe_lidar_distance: float = 8.0,
        pursuer_speed_mps: float = 10.0,
        robot_name: str | None = None,
        still_distance_threshold: float = 0.05,
        action_repeat: int = 2,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.capture_distance = capture_distance
        self.safe_lidar_distance = safe_lidar_distance
        self.pursuer_speed_mps = pursuer_speed_mps
        self.robot_name = robot_name or os.environ.get("WEBOTS_ROBOT_NAME") or "evader"
        self.still_distance_threshold = still_distance_threshold
        self.action_repeat = action_repeat

        self.action_space = spaces.Box(
            low=np.array([-0.55, -1.0], dtype=np.float32),
            high=np.array([0.55, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(0.0, 1.0, shape=(16,), dtype=np.float32),
                "pursuer": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "ego": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )

        self.driver: Any = None
        self.lidar: Any = None
        self.gps: Any = None
        self.touch_sensor: Any = None
        self.evader_node: Any = None
        self.pursuer_node: Any = None
        self.evader_translation_field: Any = None
        self.evader_rotation_field: Any = None
        self.pursuer_translation_field: Any = None
        self.pursuer_rotation_field: Any = None

        self.timestep = 32
        self.step_count = 0
        self.previous_distance = 0.0
        self.previous_position: np.ndarray | None = None
        self.current_steering = 0.0

        self.spawn_poses = (
            SpawnPose((-45.0, 46.28), (-22.98, 45.88), math.pi),
            SpawnPose((45.0, -45.0), (25.0, -45.0), 0.0),
            SpawnPose((-105.0, 4.5), (-85.0, 4.5), math.pi),
            SpawnPose((105.0, 93.0), (85.0, 93.0), 0.0),
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._ensure_webots()
        self.step_count = 0
        self.current_steering = 0.0

        pose = self.spawn_poses[0]
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

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(0.0)
        self.driver.setThrottle(0.0)
        self.driver.setBrakeIntensity(0.0)
        self.driver.setSteeringAngle(0.0)
        self._step_simulation(20)

        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        self.previous_position = evader_xy
        self.previous_distance = self._distance(evader_xy, pursuer_xy)
        return self._observation(), {"distance_to_pursuer": self.previous_distance}

    def step(self, action: np.ndarray):
        self._ensure_webots()
        self.step_count += 1

        steering = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        drive = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))
        self.current_steering = steering
        throttle = self._throttle_from_action(drive)
        target_speed = 25.0 + 35.0 * throttle

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(target_speed)
        self.driver.setSteeringAngle(steering)
        self.driver.setThrottle(throttle)
        self.driver.setBrakeIntensity(0.0)

        self._move_pursuer()
        self._step_simulation(self.action_repeat)

        obs = self._observation()
        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()
        distance = self._distance(evader_xy, pursuer_xy)
        min_lidar = float(np.min(obs["lidar"]))

        moved_distance = self._moved_distance(evader_xy)
        reward = self._reward(distance, min_lidar, drive, moved_distance)
        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        if distance <= self.capture_distance:
            reward -= 100.0
            terminated = True
        if self._has_collision(min_lidar):
            reward -= 75.0
            terminated = True

        self.previous_distance = distance
        self.previous_position = evader_xy
        info = {
            "distance_to_pursuer": distance,
            "min_lidar": min_lidar,
            "captured": distance <= self.capture_distance,
            "touch_contact": self._has_touch_contact(),
            "moved_distance": moved_distance,
            "target_speed": target_speed,
            "throttle": throttle,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.driver is not None:
            self.driver.setThrottle(0.0)
            self.driver.setBrakeIntensity(1.0)

    def _ensure_webots(self) -> None:
        if self.driver is not None:
            return

        _configure_webots_paths()
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
        self.lidar = self._device_by_name(("lidar", "evader lidar", "Lidar"))
        self.gps = self._device_by_name(("gps", "GPS"))
        self.touch_sensor = self._device_by_name(("touch sensor", "evader touch sensor", "TouchSensor"))

        if self.lidar is not None:
            self.lidar.enable(self.timestep)
        if self.gps is not None:
            self.gps.enable(self.timestep)
        if self.touch_sensor is not None:
            self.touch_sensor.enable(self.timestep)

        self.evader_node = self._node_by_def("Evader")
        self.pursuer_node = self._node_by_def("Pursuer")
        if self.evader_node is not None:
            self.evader_translation_field = self.evader_node.getField("translation")
            self.evader_rotation_field = self.evader_node.getField("rotation")
        if self.pursuer_node is not None:
            self.pursuer_translation_field = self.pursuer_node.getField("translation")
            self.pursuer_rotation_field = self.pursuer_node.getField("rotation")

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
        if node is not None:
            node.resetPhysics()

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
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        bearing = math.atan2(float(delta[1]), float(delta[0]))

        speed_kmh = float(self.driver.getCurrentSpeed()) if self.driver is not None else 0.0
        return {
            "lidar": self._lidar_bins(),
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
                    np.clip(self.current_steering / 0.55, -1.0, 1.0),
                    1.0 if self._has_touch_contact() else 0.0,
                ],
                dtype=np.float32,
            ),
        }

    def _lidar_bins(self) -> np.ndarray:
        if self.lidar is None:
            return np.ones(16, dtype=np.float32)

        ranges = np.asarray(self.lidar.getRangeImage(), dtype=np.float32)
        if ranges.size == 0:
            return np.ones(16, dtype=np.float32)

        max_range = float(self.lidar.getMaxRange())
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, max_range)
        chunks = np.array_split(ranges, 16)
        return np.array([np.min(chunk) / max_range for chunk in chunks], dtype=np.float32)

    def _reward(self, distance: float, min_lidar: float, drive: float, moved_distance: float) -> float:
        distance_delta = distance - self.previous_distance
        separation_reward = 2.5 * distance_delta
        distance_margin = np.clip((distance - self.capture_distance) / 60.0, -1.0, 1.0)
        obstacle_penalty = -3.0 * max(0.0, (self.safe_lidar_distance / 100.0) - min_lidar)
        still_penalty = -0.5 if moved_distance < self.still_distance_threshold else 0.0
        steering_penalty = -0.03 * abs(self.current_steering)
        drive_reward = 0.02 * max(0.0, drive)
        survival_reward = 0.05
        return float(
            separation_reward
            + distance_margin
            + obstacle_penalty
            + still_penalty
            + steering_penalty
            + drive_reward
            + survival_reward
        )

    @staticmethod
    def _throttle_from_action(drive: float) -> float:
        return float(np.clip(0.35 + 0.65 * ((drive + 1.0) / 2.0), 0.35, 1.0))

    def _moved_distance(self, evader_xy: np.ndarray) -> float:
        if self.previous_position is None:
            return 0.0
        return float(np.linalg.norm(evader_xy - self.previous_position))

    def _has_collision(self, min_lidar: float) -> bool:
        return self._has_touch_contact() or min_lidar <= 0.015

    def _has_touch_contact(self) -> bool:
        if self.touch_sensor is None:
            return False
        try:
            return bool(self.touch_sensor.getValue() > 0.0)
        except Exception:
            return False

    def _evader_xy(self) -> np.ndarray:
        if self.gps is not None:
            values = self.gps.getValues()
            return np.array([values[0], values[1]], dtype=np.float32)
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
        return float(np.linalg.norm(a - b))


if "Evader-v0" not in registry:
    register(
        id="Evader-v0",
        entry_point="controllers.evader_env:EvaderEnv",
    )

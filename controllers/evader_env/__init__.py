import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import cv2
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
            except:
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
    metadata = {"render_modes": []}

    def __init__(
            self,
            max_episode_steps: int = 2500,
            capture_distance: float = 6.0,
            safe_lidar_distance: float = 8.0,
            pursuer_speed_mps: float = 20.0,
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
                "ego": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )

        self.driver: Any = None
        self.lidar: Any = None
        self.gps: Any = None
        self.touch_sensor: Any = None
        self.camera_front: Any = None
        self.camera_rear: Any = None

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
        self.current_lane_offset = 0.0
        self.stuck_steps = 0

        self.spawn_poses = (
            SpawnPose((45.0, -45.0), (25.0, -45.0), 0.0),
            SpawnPose((-45.0, 46.28), (-22.98, 45.88), math.pi),
            SpawnPose((-105.0, 4.5), (-85.0, 4.5), math.pi),
            SpawnPose((105.0, 93.0), (85.0, 93.0), 0.0),
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._ensure_webots()
        self.step_count = 0
        self.current_steering = 0.0
        self.current_lane_offset = 0.0
        self.stuck_steps = 0

        pose = self.spawn_poses[0]
        ruido_angular = float(np.random.uniform(-0.15, 0.15))

        if self.evader_translation_field is not None:
            self._set_vehicle_pose(self.evader_node, self.evader_translation_field, self.evader_rotation_field,
                                   pose.evader_xy, pose.heading + ruido_angular, z=0.31)
        if self.pursuer_translation_field is not None:
            self._set_vehicle_pose(self.pursuer_node, self.pursuer_translation_field, self.pursuer_rotation_field,
                                   pose.pursuer_xy, pose.heading, z=0.55)

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(0.0)
        self.driver.setThrottle(0.0)
        self.driver.setBrakeIntensity(0.0)
        self.driver.setSteeringAngle(0.0)
        self._step_simulation(20)

        evader_xy = self._evader_xy()
        self.previous_position = evader_xy
        self.previous_distance = self._distance(evader_xy, self._pursuer_xy())
        self.current_lane_offset = self._get_lane_offset()

        return self._observation(), {"distance_to_pursuer": self.previous_distance}

    def step(self, action: np.ndarray):
        self._ensure_webots()
        self.step_count += 1

        steering = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        drive = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))
        self.current_steering = steering
        throttle = float(np.clip(0.35 + 0.65 * ((drive + 1.0) / 2.0), 0.35, 1.0))
        target_speed = 25.0 + 35.0 * throttle

        self.driver.setGear(1)
        self.driver.setCruisingSpeed(target_speed)
        self.driver.setSteeringAngle(steering)
        self.driver.setThrottle(throttle)
        self.driver.setBrakeIntensity(0.0)

        self._step_simulation(self.action_repeat)
        self.current_lane_offset = self._get_lane_offset()

        # --- LÓGICA DO PERSEGUIDOR (INÍCIO) ---
        evader_xy = self._evader_xy()
        pursuer_xy = self._pursuer_xy()

        # 1. Calculamos o tempo de cada step e o tempo total decorrido desde o reset
        dt = (self.timestep / 1000.0) * self.action_repeat
        elapsed_time = self.step_count * dt

        if self.pursuer_translation_field and self.pursuer_speed_mps > 0:
            # 2. O carro só começa a perseguir SE tiverem passado 3 ou mais segundos
            if elapsed_time >= 7.0:
                delta = evader_xy - pursuer_xy
                dist = float(np.linalg.norm(delta))

                # Só move se estiver a uma distância segura para evitar sobreposição total
                if dist > 0.5:
                    direction = delta / dist

                    # Calcula a nova posição
                    new_x = pursuer_xy[0] + (direction[0] * self.pursuer_speed_mps * dt)
                    new_y = pursuer_xy[1] + (direction[1] * self.pursuer_speed_mps * dt)

                    # Calcula a rotação para o carro apontar para o evader
                    heading = math.atan2(direction[1], direction[0])

                    # Atualiza a posição no Webots
                    self._set_vehicle_pose(
                        self.pursuer_node,
                        self.pursuer_translation_field,
                        self.pursuer_rotation_field,
                        (new_x, new_y),
                        heading,
                        z=0.55
                    )
        # --- LÓGICA DO PERSEGUIDOR (FIM) ---

        obs = self._observation()
        distance = self._distance(evader_xy, self._pursuer_xy())
        min_lidar = float(np.min(obs["lidar"]))
        moved_distance = float(
            np.linalg.norm(evader_xy - self.previous_position)) if self.previous_position is not None else 0.0

        if moved_distance < self.still_distance_threshold:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        reward = self._reward(distance, min_lidar, drive, moved_distance)
        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        if evader_xy[0] < -176.0 or evader_xy[0] > 190.0 or evader_xy[1] < -210.0 or evader_xy[1] > 275.0:
            reward -= 1000.0
            terminated = True

        if (self.touch_sensor and self.touch_sensor.getValue() > 0.0) or min_lidar <= 0.015:
            reward -= 1000.0
            terminated = True

        if self.stuck_steps > 200 and self.step_count > 150:
            reward -= 1000.0
            terminated = True

        self.previous_distance = distance
        self.previous_position = evader_xy
        return obs, reward, terminated, truncated, {"distance": distance, "lane_offset": self.current_lane_offset}

    def close(self) -> None:
        if self.driver is not None:
            self.driver.setThrottle(0.0)
            self.driver.setBrakeIntensity(1.0)

    def _get_lane_offset(self) -> float:
        if self.camera_front is None: return 0.0
        raw_image = self.camera_front.getImage()
        if not raw_image: return 0.0

        width = self.camera_front.getWidth()
        height = self.camera_front.getHeight()
        img = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        chao = img[int(height / 2):, :]

        hsv = cv2.cvtColor(cv2.cvtColor(chao, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))

        cv2.namedWindow("O Cerebro do Carro (Mascara)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("O Cerebro do Carro (Mascara)", 800, 400)
        cv2.imshow("O Cerebro do Carro (Mascara)", mask)
        cv2.waitKey(1)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            return float(np.clip(((int(M["m10"] / M["m00"])) - (width / 2.0)) / (width / 2.0), -1.0, 1.0))
        return 1.0

    def _device_by_name(self, names: tuple[str, ...]):
        for name in names:
            try:
                dev = self.driver.getDevice(name)
                if dev is not None: return dev
            except Exception:
                continue
        return None

    def _ensure_webots(self) -> None:
        if self.driver is not None: return
        _configure_webots_paths()
        os.environ["WEBOTS_ROBOT_NAME"] = self.robot_name
        from vehicle import Driver
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())

        self.lidar = self._device_by_name(("lidar", "evader lidar", "Lidar"))
        self.gps = self._device_by_name(("gps", "GPS"))
        self.touch_sensor = self._device_by_name(("touch sensor", "evader touch sensor", "TouchSensor"))
        self.camera_front = self._device_by_name(("camera", "front_camera", "Camera"))

        if self.lidar: self.lidar.enable(self.timestep)
        if self.gps: self.gps.enable(self.timestep)
        if self.touch_sensor: self.touch_sensor.enable(self.timestep)
        if self.camera_front: self.camera_front.enable(self.timestep)

        self.evader_node = self.driver.getFromDef("Evader")
        self.pursuer_node = self.driver.getFromDef("Pursuer")
        if self.evader_node:
            self.evader_translation_field = self.evader_node.getField("translation")
            self.evader_rotation_field = self.evader_node.getField("rotation")
        if self.pursuer_node:
            self.pursuer_translation_field = self.pursuer_node.getField("translation")
            self.pursuer_rotation_field = self.pursuer_node.getField("rotation")

    def _step_simulation(self, substeps: int) -> None:
        for _ in range(substeps):
            if self.driver.step() == -1: break

    def _set_vehicle_pose(self, node: Any, translation_field: Any, rotation_field: Any, xy: tuple[float, float],
                          heading: float, z: float) -> None:
        translation_field.setSFVec3f([xy[0], xy[1], z])
        rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])
        if node: node.resetPhysics()

    def _observation(self) -> dict[str, np.ndarray]:
        evader_xy = self._evader_xy()
        delta = self._pursuer_xy() - evader_xy
        dist = max(float(np.linalg.norm(delta)), 1e-6)
        speed = float(self.driver.getCurrentSpeed()) if self.driver else 0.0
        return {
            "lidar": self._lidar_bins(),
            "pursuer": np.array(
                [delta[0] / 120.0, delta[1] / 120.0, dist / 120.0, math.sin(math.atan2(delta[1], delta[0]))],
                dtype=np.float32),
            "ego": np.array([speed / 70.0, self.current_steering / 0.55,
                             1.0 if (self.touch_sensor and self.touch_sensor.getValue() > 0) else 0.0,
                             self.current_lane_offset], dtype=np.float32),
        }

    def _lidar_bins(self) -> np.ndarray:
        if not self.lidar: return np.ones(16, dtype=np.float32)
        raw = self.lidar.getRangeImage()
        if not raw: return np.ones(16, dtype=np.float32)
        max_r = float(self.lidar.getMaxRange())
        ranges = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=max_r, posinf=max_r, neginf=0.0)
        return np.array([np.min(c) / max_r for c in np.array_split(np.clip(ranges, 0, max_r), 16)], dtype=np.float32)

    def _reward(self, distance: float, min_lidar: float, drive: float, moved_distance: float) -> float:
        still_penalty = -3.0 if moved_distance < self.still_distance_threshold else 0.0
        obstacle_penalty = -3.0 * max(0.0, (self.safe_lidar_distance / 100.0) - min_lidar)

        lane_error = abs(self.current_lane_offset)

        if lane_error < 0.6:
            lane_reward = 3.0 * (1.0 - lane_error)
            drive_reward = 5.0 * moved_distance
            survival_bonus = 0.05
            steering_penalty = -2.0 * moved_distance * (self.current_steering ** 2)
        else:
            lane_reward = -2.0
            drive_reward = 0.0
            survival_bonus = 0.0
            steering_penalty = 0.0

        return float(
            obstacle_penalty
            + still_penalty
            + steering_penalty
            + lane_reward
            + drive_reward
            + survival_bonus
        )

    def _evader_xy(self) -> np.ndarray:
        if not self.gps: return np.zeros(2, dtype=np.float32)
        v = self.gps.getValues()
        return np.array([v[0], v[1]], dtype=np.float32)

    def _pursuer_xy(self) -> np.ndarray:
        if not self.pursuer_translation_field: return np.zeros(2, dtype=np.float32)
        v = self.pursuer_translation_field.getSFVec3f()
        return np.array([v[0], v[1]], dtype=np.float32)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))


if "Evader-v0" not in registry:
    register(id="Evader-v0", entry_point="controllers.evader_env:EvaderEnv")
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_WEBOTS_HOME = r"C:\Program Files\Webots"
CRUISING_SPEED_KMH = 40.0
THROTTLE = 0.8
STEERING_ANGLE = 0.0
DEBUG_PRINT_EVERY_STEPS = 20
OBSTACLE_NEAR_THRESHOLD_M = 6.0


def configure_webots_environment() -> str:
    webots_home = os.environ.get("WEBOTS_HOME", DEFAULT_WEBOTS_HOME)
    os.environ["WEBOTS_HOME"] = webots_home

    dll_paths = [
        os.path.join(webots_home, "msys64", "mingw64", "bin"),
        os.path.join(webots_home, "msys64", "mingw64", "bin", "cpp"),
        os.path.join(webots_home, "lib", "controller"),
        os.path.join(webots_home, "projects", "vehicles", "lib"),
    ]

    for path in dll_paths:
        if not os.path.exists(path):
            continue
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    controller_python_path = os.path.join(webots_home, "lib", "controller", "python")
    if controller_python_path not in sys.path:
        sys.path.append(controller_python_path)

    return webots_home


configure_webots_environment()

from vehicle import Driver


@dataclass
class EvaderObservation:
    front_distance: float
    left_distance: float
    right_distance: float
    obstacle_near: bool
    speed_kmh: float
    heading_rad: Optional[float]
    yaw_rate: Optional[float]
    camera_width: int
    camera_height: int
    camera_frame_ok: bool
    lidar_active: bool
    camera_active: bool
    gps_active: bool
    gyro_active: bool

    def debug_summary(self) -> str:
        heading = "n/a" if self.heading_rad is None else f"{self.heading_rad:.2f}rad"
        yaw_rate = "n/a" if self.yaw_rate is None else f"{self.yaw_rate:.2f}"
        return (
            "OBS "
            f"front={self.front_distance:.2f}m "
            f"left={self.left_distance:.2f}m "
            f"right={self.right_distance:.2f}m "
            f"near={self.obstacle_near} "
            f"speed={self.speed_kmh:.2f}km/h "
            f"heading={heading} "
            f"yaw_rate={yaw_rate} "
            f"camera={self.camera_width}x{self.camera_height} "
            f"frame_ok={self.camera_frame_ok} "
            f"lidar={self.lidar_active} "
            f"gps={self.gps_active} "
            f"gyro={self.gyro_active}"
        )


class SensorSuite:
    def __init__(self, driver: Driver, timestep: int):
        self.driver = driver
        self.timestep = timestep
        self.devices = self._discover_devices()

        self.camera = self._pick_device("evader_camera", "camera")
        self.lidar = self._pick_device("evader_lidar", "lidar")
        self.gps = self._pick_device("evader_gps", "gps")
        self.gyro = self._pick_device("evader_gyro", "gyro")

        self.previous_gps_xy = None

    def _discover_devices(self) -> Dict[str, object]:
        devices = {}
        device_names = []

        for index in range(self.driver.getNumberOfDevices()):
            device = self.driver.getDeviceByIndex(index)
            if device is None:
                continue
            name = device.getName()
            devices[name] = device
            device_names.append(name)

        print("Detected Webots devices:", ", ".join(device_names) if device_names else "none")
        return devices

    def _pick_device(self, *names: str):
        for name in names:
            if name in self.devices:
                return self.devices[name]
        return None

    def enable_available_sensors(self) -> None:
        self._enable_sensor("Camera", self.camera)
        self._enable_sensor("Lidar", self.lidar)
        self._enable_sensor("GPS", self.gps)
        self._enable_sensor("Gyro", self.gyro)

    def _enable_sensor(self, label: str, sensor) -> None:
        if sensor is None:
            print(f"{label}: not found")
            return
        if hasattr(sensor, "enable"):
            sensor.enable(self.timestep)
            print(f"{label}: enabled ({sensor.getName()})")
        else:
            print(f"{label}: found but does not support enable() ({sensor.getName()})")

    def build_observation(self, speed_kmh: float) -> EvaderObservation:
        lidar_summary = self._read_lidar_summary()
        camera_summary = self._read_camera_summary()
        heading_rad = self._estimate_heading_from_gps()
        yaw_rate = self._read_yaw_rate()

        return EvaderObservation(
            front_distance=lidar_summary["front_distance"],
            left_distance=lidar_summary["left_distance"],
            right_distance=lidar_summary["right_distance"],
            obstacle_near=lidar_summary["obstacle_near"],
            speed_kmh=speed_kmh,
            heading_rad=heading_rad,
            yaw_rate=yaw_rate,
            camera_width=camera_summary["width"],
            camera_height=camera_summary["height"],
            camera_frame_ok=camera_summary["frame_ok"],
            lidar_active=lidar_summary["active"],
            camera_active=camera_summary["active"],
            gps_active=self.gps is not None,
            gyro_active=self.gyro is not None,
        )

    def _read_camera_summary(self) -> Dict[str, object]:
        if self.camera is None:
            return {"active": False, "width": 0, "height": 0, "frame_ok": False}

        frame = self.camera.getImage()
        return {
            "active": True,
            "width": self.camera.getWidth(),
            "height": self.camera.getHeight(),
            "frame_ok": frame is not None and len(frame) > 0,
        }

    def _read_lidar_summary(self) -> Dict[str, object]:
        if self.lidar is None:
            return {
                "active": False,
                "front_distance": math.inf,
                "left_distance": math.inf,
                "right_distance": math.inf,
                "obstacle_near": False,
            }

        range_image = list(self.lidar.getRangeImage())
        resolution = len(range_image)
        if resolution == 0:
            return {
                "active": True,
                "front_distance": math.inf,
                "left_distance": math.inf,
                "right_distance": math.inf,
                "obstacle_near": False,
            }

        valid_ranges = [value for value in range_image if math.isfinite(value) and value > 0.0]
        if not valid_ranges:
            return {
                "active": True,
                "front_distance": math.inf,
                "left_distance": math.inf,
                "right_distance": math.inf,
                "obstacle_near": False,
            }

        side_window = max(1, resolution // 3)
        front_window = max(1, resolution // 6)
        center = resolution // 2

        left_slice = range_image[:side_window]
        front_slice = range_image[max(0, center - front_window) : min(resolution, center + front_window)]
        right_slice = range_image[-side_window:]

        left_distance = self._safe_min_distance(left_slice)
        front_distance = self._safe_min_distance(front_slice)
        right_distance = self._safe_min_distance(right_slice)
        nearest_obstacle = min(front_distance, left_distance, right_distance)

        return {
            "active": True,
            "front_distance": front_distance,
            "left_distance": left_distance,
            "right_distance": right_distance,
            "obstacle_near": nearest_obstacle < OBSTACLE_NEAR_THRESHOLD_M,
        }

    def _safe_min_distance(self, values) -> float:
        valid = [value for value in values if math.isfinite(value) and value > 0.0]
        return min(valid) if valid else math.inf

    def _estimate_heading_from_gps(self) -> Optional[float]:
        if self.gps is None:
            return None

        gps_values = self.gps.getValues()
        if gps_values is None or len(gps_values) < 2:
            return None

        current_xy = (gps_values[0], gps_values[1])
        if self.previous_gps_xy is None:
            self.previous_gps_xy = current_xy
            return None

        delta_x = current_xy[0] - self.previous_gps_xy[0]
        delta_y = current_xy[1] - self.previous_gps_xy[1]
        self.previous_gps_xy = current_xy

        if abs(delta_x) < 1e-4 and abs(delta_y) < 1e-4:
            return None

        return math.atan2(delta_y, delta_x)

    def _read_yaw_rate(self) -> Optional[float]:
        if self.gyro is None:
            return None

        gyro_values = self.gyro.getValues()
        if gyro_values is None or len(gyro_values) < 3:
            return None

        return gyro_values[2]


def apply_basic_motion(driver: Driver) -> None:
    driver.setGear(1)
    driver.setCruisingSpeed(CRUISING_SPEED_KMH)
    driver.setThrottle(THROTTLE)
    driver.setBrakeIntensity(0.0)
    driver.setSteeringAngle(STEERING_ANGLE)


def run_evader() -> None:
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())
    sensors = SensorSuite(driver, timestep)
    sensors.enable_available_sensors()

    print("=====================================================")
    print("EVADER: controller connected and sensor check enabled.")
    print("=====================================================")

    apply_basic_motion(driver)

    step_count = 0
    while driver.step() != -1:
        step_count += 1
        current_speed = driver.getCurrentSpeed()
        observation = sensors.build_observation(current_speed)

        if step_count % DEBUG_PRINT_EVERY_STEPS == 0:
            print(observation.debug_summary())
            if observation.obstacle_near:
                print("ALERT obstacle_near=True")

        apply_basic_motion(driver)


if __name__ == "__main__":
    run_evader()

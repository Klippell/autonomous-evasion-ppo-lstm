import json
import math
from typing import Optional, Tuple

from vehicle import Driver


CRUISING_SPEED_KMH = 55.0
THROTTLE = 0.55
MAX_STEERING_ANGLE = 0.45
STEERING_GAIN = 0.75
DEBUG_PRINT_EVERY_STEPS = 25


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PursuerController:
    def __init__(self) -> None:
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        self.gps = self.driver.getDevice("pursuer_gps")
        self.receiver = self.driver.getDevice("pursuer_receiver")

        self.gps.enable(self.timestep)
        self.receiver.enable(self.timestep)

        self.last_known_evader_position: Optional[Tuple[float, float]] = None
        self.previous_xy: Optional[Tuple[float, float]] = None
        self.heading_rad: Optional[float] = None
        self.episode_id = 0

    def run(self) -> None:
        self._apply_motion(0.0)

        step_count = 0
        while self.driver.step() != -1:
            step_count += 1
            self._consume_messages()
            current_xy = self._get_current_xy()
            self._update_heading(current_xy)
            steering_angle = self._compute_steering(current_xy)
            self._apply_motion(steering_angle)

            if step_count % DEBUG_PRINT_EVERY_STEPS == 0:
                target = "none"
                if self.last_known_evader_position is not None:
                    target = (
                        f"({self.last_known_evader_position[0]:.2f}, "
                        f"{self.last_known_evader_position[1]:.2f})"
                    )
                heading = "n/a" if self.heading_rad is None else f"{self.heading_rad:.2f}"
                print(
                    "PURSUER "
                    f"episode={self.episode_id} "
                    f"target={target} "
                    f"heading={heading} "
                    f"steering={steering_angle:.2f}"
                )

    def _consume_messages(self) -> None:
        while self.receiver.getQueueLength() > 0:
            raw_message = bytes(self.receiver.getBytes()).decode("utf-8")
            self.receiver.nextPacket()
            if not raw_message:
                continue

            payload = json.loads(raw_message)
            incoming_episode_id = payload.get("episode_id", self.episode_id)
            if incoming_episode_id != self.episode_id:
                self.episode_id = incoming_episode_id
                self.previous_xy = None
                self.heading_rad = None

            target_position = payload.get("last_known_evader_position")
            if target_position is not None:
                self.last_known_evader_position = (target_position[0], target_position[1])

    def _get_current_xy(self) -> Optional[Tuple[float, float]]:
        values = self.gps.getValues()
        if values is None or len(values) < 2:
            return None
        return values[0], values[1]

    def _update_heading(self, current_xy: Optional[Tuple[float, float]]) -> None:
        if current_xy is None:
            return
        if self.previous_xy is None:
            self.previous_xy = current_xy
            return

        delta_x = current_xy[0] - self.previous_xy[0]
        delta_y = current_xy[1] - self.previous_xy[1]
        self.previous_xy = current_xy

        if abs(delta_x) < 1e-4 and abs(delta_y) < 1e-4:
            return

        self.heading_rad = math.atan2(delta_y, delta_x)

    def _compute_steering(self, current_xy: Optional[Tuple[float, float]]) -> float:
        if current_xy is None or self.last_known_evader_position is None or self.heading_rad is None:
            return 0.0

        target_dx = self.last_known_evader_position[0] - current_xy[0]
        target_dy = self.last_known_evader_position[1] - current_xy[1]
        target_heading = math.atan2(target_dy, target_dx)
        heading_error = normalize_angle(target_heading - self.heading_rad)
        return clamp(STEERING_GAIN * heading_error, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    def _apply_motion(self, steering_angle: float) -> None:
        self.driver.setGear(1)
        self.driver.setCruisingSpeed(CRUISING_SPEED_KMH)
        self.driver.setThrottle(THROTTLE)
        self.driver.setBrakeIntensity(0.0)
        self.driver.setSteeringAngle(steering_angle)


if __name__ == "__main__":
    PursuerController().run()

import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from controller import Supervisor


GPS_UPDATE_PERIOD_S = 2.0
CAPTURE_DISTANCE_M = 3.0
CAPTURE_HOLD_TIME_S = 1.0
TIMEOUT_S = 45.0
COLLISION_MIN_EPISODE_AGE_S = 5.0
COLLISION_STOP_SPEED_MPS = 0.15
COLLISION_STOP_HOLD_S = 2.0
DEBUG_PRINT_EVERY_STEPS = 20


def planar_xy(position_3d) -> Tuple[float, float]:
    return position_3d[0], position_3d[1]


def planar_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class EpisodeState:
    episode_id: int = 1
    elapsed_s: float = 0.0
    capture_hold_s: float = 0.0
    stalled_hold_s: float = 0.0
    done: bool = False
    done_reason: Optional[str] = None
    last_known_evader_position: Optional[Tuple[float, float]] = None
    last_gps_update_s: float = -1e9

    def reset(self) -> None:
        self.episode_id += 1
        self.elapsed_s = 0.0
        self.capture_hold_s = 0.0
        self.stalled_hold_s = 0.0
        self.done = False
        self.done_reason = None
        self.last_known_evader_position = None
        self.last_gps_update_s = -1e9


class EpisodeSupervisor:
    def __init__(self) -> None:
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self.evader_node = self.supervisor.getFromDef("EVADER")
        self.pursuer_node = self.supervisor.getFromDef("PURSUER")
        self.emitter = self.supervisor.getDevice("episode_emitter")

        if self.evader_node is None or self.pursuer_node is None:
            raise RuntimeError("Supervisor could not find EVADER or PURSUER by DEF.")

        self.evader_translation_field = self.evader_node.getField("translation")
        self.evader_rotation_field = self.evader_node.getField("rotation")
        self.pursuer_translation_field = self.pursuer_node.getField("translation")
        self.pursuer_rotation_field = self.pursuer_node.getField("rotation")

        self.evader_spawn_translation = list(self.evader_translation_field.getSFVec3f())
        self.evader_spawn_rotation = list(self.evader_rotation_field.getSFRotation())
        self.pursuer_spawn_translation = list(self.pursuer_translation_field.getSFVec3f())
        self.pursuer_spawn_rotation = list(self.pursuer_rotation_field.getSFRotation())

        self.state = EpisodeState()

        print("SUPERVISOR horizontal_plane=x/y z_is_height=True")

    def run(self) -> None:
        step_count = 0
        while self.supervisor.step(self.timestep) != -1:
            step_count += 1
            self.state.elapsed_s += self.dt

            evader_xy = planar_xy(self.evader_node.getPosition())
            pursuer_xy = planar_xy(self.pursuer_node.getPosition())
            distance = planar_distance(evader_xy, pursuer_xy)

            self._maybe_publish_evader_position(evader_xy)
            self._update_capture(distance)
            self._update_collision(distance)
            self._update_timeout()

            if step_count % DEBUG_PRINT_EVERY_STEPS == 0:
                target = "none"
                if self.state.last_known_evader_position is not None:
                    target = (
                        f"({self.state.last_known_evader_position[0]:.2f}, "
                        f"{self.state.last_known_evader_position[1]:.2f})"
                    )
                print(
                    "EPISODE "
                    f"id={self.state.episode_id} "
                    f"elapsed={self.state.elapsed_s:.1f}s "
                    f"evader=({evader_xy[0]:.2f},{evader_xy[1]:.2f}) "
                    f"pursuer=({pursuer_xy[0]:.2f},{pursuer_xy[1]:.2f}) "
                    f"target={target} "
                    f"distance={distance:.2f}m "
                    f"capture_hold={self.state.capture_hold_s:.2f}s "
                    f"status={'done:' + self.state.done_reason if self.state.done else 'running'}"
                )

            if self.state.done:
                print(f"EPISODE_END id={self.state.episode_id} reason={self.state.done_reason}")
                self._reset_episode()

    def _maybe_publish_evader_position(self, evader_xy: Tuple[float, float]) -> None:
        if self.state.elapsed_s - self.state.last_gps_update_s < GPS_UPDATE_PERIOD_S:
            return

        self.state.last_gps_update_s = self.state.elapsed_s
        self.state.last_known_evader_position = evader_xy
        self._send_target_update()

    def _send_target_update(self) -> None:
        payload = {
            "episode_id": self.state.episode_id,
            "last_known_evader_position": list(self.state.last_known_evader_position)
            if self.state.last_known_evader_position is not None
            else None,
        }
        self.emitter.send(json.dumps(payload).encode("utf-8"))

    def _update_capture(self, distance: float) -> None:
        if self.state.done:
            return

        if distance <= CAPTURE_DISTANCE_M:
            self.state.capture_hold_s += self.dt
        else:
            self.state.capture_hold_s = 0.0

        if self.state.capture_hold_s >= CAPTURE_HOLD_TIME_S:
            self.state.done = True
            self.state.done_reason = "captured"

    def _update_collision(self, distance: float) -> None:
        if self.state.done:
            return

        if self.state.elapsed_s < COLLISION_MIN_EPISODE_AGE_S:
            self.state.stalled_hold_s = 0.0
            return

        evader_velocity = self.evader_node.getVelocity()
        linear_speed = math.hypot(evader_velocity[0], evader_velocity[1])

        if linear_speed < COLLISION_STOP_SPEED_MPS and distance > CAPTURE_DISTANCE_M:
            self.state.stalled_hold_s += self.dt
        else:
            self.state.stalled_hold_s = 0.0

        if self.state.stalled_hold_s >= COLLISION_STOP_HOLD_S:
            self.state.done = True
            self.state.done_reason = "collision"

    def _update_timeout(self) -> None:
        if self.state.done:
            return

        if self.state.elapsed_s >= TIMEOUT_S:
            self.state.done = True
            self.state.done_reason = "timeout"

    def _reset_episode(self) -> None:
        self.evader_translation_field.setSFVec3f(self.evader_spawn_translation)
        self.evader_rotation_field.setSFRotation(self.evader_spawn_rotation)
        self.pursuer_translation_field.setSFVec3f(self.pursuer_spawn_translation)
        self.pursuer_rotation_field.setSFRotation(self.pursuer_spawn_rotation)

        self.evader_node.resetPhysics()
        self.pursuer_node.resetPhysics()

        self.state.reset()
        self.state.last_known_evader_position = planar_xy(self.evader_spawn_translation)
        self._send_target_update()
        print(f"EPISODE_RESET id={self.state.episode_id}")


if __name__ == "__main__":
    EpisodeSupervisor().run()

"""Debug drawing helpers for the evader Gym environment."""
from __future__ import annotations

import math

import numpy as np


class DebugDisplayMixin:
    def _draw_reward_label(self, reward: float, info: dict[str, float | bool]) -> None:
        if not self.show_reward_display:
            return

        color = 0x22FF88 if reward >= 0.0 else 0xFF5555
        text = (
            f"reward {reward:+.2f} | "
            f"pursuer {info['distance_to_pursuer']:.1f} m | "
            f"front {info['front_obstacle_distance']:.1f} m | "
            f"los {int(info['pursuer_visible'])} F/B {int(info['front_pursuer_visible'])}/{int(info['back_pursuer_visible'])} | "
            f"L/R/B {info['left_obstacle_distance']:.1f}/{info['right_obstacle_distance']:.1f}/{info['back_obstacle_distance']:.1f} | "
            f"T/B {info['throttle']:.2f}/{info['brake']:.2f} | "
            f"move {info['movement_reward'] + info['still_penalty'] + info['survival_reward'] + info['exploration_reward']:+.2f} | "
            f"cam {info['visual_moving_away_reward']:+.2f} | "
            f"stab {info['stability_reward_total']:+.2f} | "
            f"obsR {info['obstacle_clearance_delta_reward'] + info['front_obstacle_penalty'] + info['side_obstacle_penalty'] + info['back_obstacle_penalty'] + info['back_approach_penalty']:+.2f}"
        )
        try:
            self.driver.setLabel(0, text, 0.02, 0.03, 0.08, color, 0.0, "Arial")
            if not self.label_debug_reported:
                print("Webots supervisor debug label is active.")
                self.label_debug_reported = True
        except Exception:
            if not self.label_debug_reported:
                print("Webots supervisor debug label failed; using terminal fallback.")
                self.label_debug_reported = True
            self._log_reward_fallback(reward, info)

    def _draw_supervisor_minimap(self, info: dict[str, float | bool]) -> None:
        if not self.show_reward_display:
            return

        origin_x = 0.80
        origin_y = 0.10
        map_size = 0.16
        meters_visible = 14.0
        center_x = origin_x + map_size * 0.5
        center_y = origin_y + map_size * 0.58
        scale = (map_size * 0.48) / meters_visible

        try:
            self.driver.setLabel(10, "PATH", origin_x, origin_y, 0.045, 0xFFFFFF, 0.0, "Arial")
            self.driver.setLabel(11, "+", center_x, center_y - map_size * 0.48, 0.035, 0x444444, 0.0, "Arial")
            self.driver.setLabel(12, "+", center_x, center_y + map_size * 0.48, 0.035, 0x444444, 0.0, "Arial")
            self.driver.setLabel(13, "+", center_x - map_size * 0.48, center_y, 0.035, 0x444444, 0.0, "Arial")
            self.driver.setLabel(14, "+", center_x + map_size * 0.48, center_y, 0.035, 0x444444, 0.0, "Arial")
            self.driver.setLabel(15, "E", center_x, center_y, 0.045, 0xFFFFFF, 0.0, "Arial")
            self.driver.setLabel(16, "|", center_x, center_y - 0.025, 0.04, 0x55AAFF, 0.0, "Arial")

            obstacles = (
                (0.0, -float(info["front_obstacle_distance"])),
                (-float(info["left_obstacle_distance"]), 0.0),
                (float(info["right_obstacle_distance"]), 0.0),
                (0.0, float(info["back_obstacle_distance"])),
            )
            for index, (local_x, local_y) in enumerate(obstacles, start=20):
                distance = math.hypot(local_x, local_y)
                if distance > meters_visible:
                    self.driver.setLabel(index, "", 0.0, 0.0, 0.01, 0xFFFFFF, 1.0, "Arial")
                    continue
                color = 0xFF4444 if distance < 5.0 else 0xFFAA33
                self.driver.setLabel(index, "X", center_x + local_x * scale, center_y + local_y * scale, 0.04, color, 0.0, "Arial")

            speed_mps = max(0.0, float(info["speed_kmh"]) / 3.6)
            steering = float(np.clip(self.current_steering, -0.55, 0.55))
            wheelbase = 2.9
            dt = 0.18
            local_x = 0.0
            local_y = 0.0
            heading = 0.0
            path_color = 0x00FFAA if float(info["path_prediction_reward"]) >= 0.0 else 0xFF5555
            for point in range(12):
                heading += (speed_mps / wheelbase) * math.tan(steering) * dt
                local_x += speed_mps * math.sin(heading) * dt
                local_y -= speed_mps * math.cos(heading) * dt
                label_id = 30 + point
                if speed_mps < 0.2 or math.hypot(local_x, local_y) > meters_visible:
                    self.driver.setLabel(label_id, "", 0.0, 0.0, 0.01, 0xFFFFFF, 1.0, "Arial")
                    continue
                self.driver.setLabel(label_id, ".", center_x + local_x * scale, center_y + local_y * scale, 0.055, path_color, 0.0, "Arial")
        except Exception:
            return

    def _draw_reward_display(self, reward: float, info: dict[str, float | bool]) -> None:
        if not self.show_reward_display or self.display is None:
            return

        try:
            width = self.display.getWidth()
            height = self.display.getHeight()
            if width <= 0 or height <= 0:
                self._log_reward_fallback(reward, info)
                return
            self.display.setColor(0x000000)
            self.display.fillRectangle(0, 0, width, height)

            self.display.setColor(0x00FFAA if reward >= 0.0 else 0xFF5555)
            self.display.setFont("Arial", 10, True)
            self.display.drawText(f"reward {reward:+.2f}", 4, 4)
            self.display.setColor(0xFFFFFF)
            self.display.drawText(f"pursuer {info['distance_to_pursuer']:.1f}m", 4, 18)
            self.display.drawText(f"path {info['path_prediction_reward']:+.2f}", 4, 32)

            self._draw_path_minimap(info, width, height)

            bar_width = max(1, width - 8)
            reward_fraction = float(np.clip((reward + 5.0) / 10.0, 0.0, 1.0))
            self.display.setColor(0x333333)
            self.display.fillRectangle(4, height - 18, bar_width, 10)
            self.display.setColor(0x00FFAA if reward >= 0.0 else 0xFF5555)
            filled_width = int(bar_width * reward_fraction)
            if filled_width > 0:
                self.display.fillRectangle(4, height - 18, filled_width, 10)
        except Exception:
            self._log_reward_fallback(reward, info)
            return

    def _draw_path_minimap(self, info: dict[str, float | bool], width: int, height: int) -> None:
        center_x = width // 2
        center_y = int(height * 0.62)
        map_radius = max(24, min(width, height) // 3)
        meters_visible = 14.0
        scale = map_radius / meters_visible

        self.display.setColor(0x222222)
        self.display.drawOval(center_x - map_radius, center_y - map_radius, map_radius * 2, map_radius * 2)
        self.display.setColor(0x444444)
        self.display.drawLine(center_x, center_y - map_radius, center_x, center_y + map_radius)
        self.display.drawLine(center_x - map_radius, center_y, center_x + map_radius, center_y)

        obstacles = (
            (0.0, -float(info["front_obstacle_distance"])),
            (-float(info["left_obstacle_distance"]), 0.0),
            (float(info["right_obstacle_distance"]), 0.0),
            (0.0, float(info["back_obstacle_distance"])),
        )
        for local_x, local_y in obstacles:
            distance = math.hypot(local_x, local_y)
            if distance > meters_visible:
                continue
            px = int(center_x + local_x * scale)
            py = int(center_y + local_y * scale)
            self.display.setColor(0xFF4444 if distance < 5.0 else 0xFFAA33)
            self.display.fillOval(px - 3, py - 3, 6, 6)

        self.display.setColor(0x55AAFF)
        self.display.drawLine(center_x, center_y - 8, center_x, center_y - 18)
        self.display.setColor(0xFFFFFF)
        self.display.fillRectangle(center_x - 3, center_y - 5, 6, 10)

        speed_mps = max(0.0, float(info["speed_kmh"]) / 3.6)
        steering = float(np.clip(self.current_steering, -0.55, 0.55))
        wheelbase = 2.9
        dt = 0.12
        local_x = 0.0
        local_y = 0.0
        heading = 0.0
        prev_x = center_x
        prev_y = center_y
        path_color = 0x00FFAA if float(info["path_prediction_reward"]) >= 0.0 else 0xFF5555
        self.display.setColor(path_color)
        for _ in range(18):
            heading += (speed_mps / wheelbase) * math.tan(steering) * dt
            local_x += speed_mps * math.sin(heading) * dt
            local_y -= speed_mps * math.cos(heading) * dt
            if math.hypot(local_x, local_y) > meters_visible:
                break
            px = int(center_x + local_x * scale)
            py = int(center_y + local_y * scale)
            self.display.drawLine(prev_x, prev_y, px, py)
            prev_x = px
            prev_y = py

    def _log_reward_fallback(self, reward: float, info: dict[str, float | bool]) -> None:
        if self.step_count - self.last_reward_log_step < 25:
            return
        self.last_reward_log_step = self.step_count
        print(
            "reward={:+.2f} dist={:.1f}m front={:.2f} moved={:.2f}m contact={}".format(
                reward,
                info["distance_to_pursuer"],
                info["front_obstacle_distance"],
                info["moved_distance"],
                info["touch_contact"],
            )
        )

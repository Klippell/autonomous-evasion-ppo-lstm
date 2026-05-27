"""Reward shaping for the evader Gym environment."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    # Pursuer / camera rewards.
    moving_away_positive_weight: float = 4.0
    moving_away_negative_weight: float = 8.0
    include_moving_away_reward: bool = False
    line_of_sight_hidden_reward: float = 1.0
    line_of_sight_break_bonus: float = 25.0
    line_of_sight_visible_penalty: float = -0.5
    visual_moving_away_weight: float = 12.0
    include_visual_moving_away_reward: bool = True
    exploration_new_cell_reward: float = 2.0
    exploration_revisit_penalty: float = -0.05

    # Obstacle rewards.
    front_obstacle_penalty_weight: float = -9.0
    side_obstacle_penalty_weight: float = -2.0
    back_obstacle_penalty_weight: float = -1.0
    front_clearance_delta_weight: float = 3.0
    side_clearance_delta_weight: float = 1.0
    back_clearance_delta_weight: float = 0.5
    back_approach_penalty_weight: float = -3.0
    obstacle_stall_penalty: float = -1.0

    # Movement / progress rewards.
    still_penalty: float = -3.0
    not_stuck_reward: float = 0.3
    movement_reward_weight: float = 4.0
    drive_reward_weight: float = 0.08
    survival_reward_weight: float = 0.1

    # Stability / control rewards.
    steering_penalty_weight: float = -0.01
    include_steering_penalty: bool = False
    fast_turn_penalty_weight: float = -1.0
    tight_turn_penalty_weight: float = -2.0
    accelerating_turn_penalty_weight: float = -6.0
    clear_front_turn_penalty_weight: float = -0.7
    straighten_reward_weight: float = 1.2
    turn_towards_obstacle_penalty_weight: float = -2.0
    clear_path_reward_weight: float = 0.4
    predicted_collision_penalty_weight: float = -8.0
    action_smoothness_penalty_weight: float = -0.3
    tilt_penalty_weight: float = -5.0
    overspeed_penalty_weight: float = -4.0


def reward_weights_from_mapping(values: dict[str, object] | RewardWeights | None) -> RewardWeights:
    if values is None:
        return RewardWeights()
    if isinstance(values, RewardWeights):
        return values
    allowed = {field.name for field in fields(RewardWeights)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown reward weight(s): {', '.join(unknown)}")
    return RewardWeights(**values)


class RewardMixin:
    def _reward(
        self,
        distance: float,
        sector_distances: dict[str, float],
        action: np.ndarray,
        speed_kmh: float,
        moved_distance: float,
    ) -> tuple[float, dict[str, float]]:
        w = self.reward_weights
        vision = self._camera_pursuer_observation()
        drive = float(action[1])
        distance_delta = distance - self.previous_distance
        if distance_delta > 0:
            moving_away_reward = w.moving_away_positive_weight * distance_delta
        else:
            moving_away_reward = w.moving_away_negative_weight * distance_delta
        line_of_sight_reward = self._line_of_sight_reward(vision)
        visual_moving_away_reward = self._visual_moving_away_reward(vision)
        exploration_reward = self._exploration_reward()
        front_obstacle_penalty = self._front_obstacle_penalty()
        side_obstacle_penalty = w.side_obstacle_penalty_weight * (
            math.exp(-sector_distances["left"] / 3.0)
            + math.exp(-sector_distances["right"] / 3.0)
        )
        back_obstacle_penalty = w.back_obstacle_penalty_weight * math.exp(-sector_distances["back"] / 2.5)
        obstacle_clearance_delta_reward = (
            w.front_clearance_delta_weight * (sector_distances["front"] - self.previous_sector_distances["front"])
            + w.side_clearance_delta_weight * (sector_distances["left"] - self.previous_sector_distances["left"])
            + w.side_clearance_delta_weight * (sector_distances["right"] - self.previous_sector_distances["right"])
            + w.back_clearance_delta_weight * (sector_distances["back"] - self.previous_sector_distances["back"])
        )
        back_distance = sector_distances["back"]
        back_approach = max(0.0, self.previous_sector_distances["back"] - back_distance)
        back_danger_range = 8.0
        back_proximity = max(0.0, (back_danger_range - back_distance) / back_danger_range)
        back_approach_penalty = w.back_approach_penalty_weight * back_approach * (1.0 + 4.0 * back_proximity)
        is_stuck = moved_distance < self.still_distance_threshold
        still_penalty = w.still_penalty if is_stuck else w.not_stuck_reward
        movement_reward = w.movement_reward_weight * moved_distance
        obstacle_stall_penalty = (
            w.obstacle_stall_penalty
            if moved_distance < self.still_distance_threshold and sector_distances["front"] < 8.0
            else 0.0
        )
        steering_penalty = w.steering_penalty_weight * abs(self.current_steering)
        normalized_speed = np.clip(speed_kmh / 70.0, 0.0, 1.0)
        fast_turn_penalty = w.fast_turn_penalty_weight * normalized_speed * abs(self.current_steering)
        steering_fraction = abs(float(np.clip(self.current_steering / 0.55, -1.0, 1.0)))
        drive_fraction = max(0.0, drive)
        tight_turn_penalty = w.tight_turn_penalty_weight * max(0.0, steering_fraction - 0.55) ** 2
        accelerating_turn_penalty = w.accelerating_turn_penalty_weight * drive_fraction * steering_fraction
        front_clearance = sector_distances["front"]
        front_clear_fraction = float(np.clip((front_clearance - 8.0) / 10.0, 0.0, 1.0))
        previous_steering_fraction = abs(float(np.clip(self.previous_action[0] / 0.55, -1.0, 1.0)))
        clear_front_turn_penalty = w.clear_front_turn_penalty_weight * steering_fraction * front_clear_fraction
        straighten_reward = (
            w.straighten_reward_weight
            * max(0.0, previous_steering_fraction - steering_fraction)
            * front_clear_fraction
        )
        turn_towards_obstacle_penalty = self._turn_towards_obstacle_penalty()
        path_prediction_reward = self._path_prediction_reward(speed_kmh, sector_distances)
        action_smoothness_penalty = w.action_smoothness_penalty_weight * float(np.linalg.norm(action - self.previous_action))
        tilt_penalty = w.tilt_penalty_weight * max(0.0, self._evader_tilt_angle() - 0.35)
        speed_mps = speed_kmh / 3.6
        overspeed = max(0.0, speed_mps - (2.5 * self.pursuer_speed_mps))
        overspeed_penalty = w.overspeed_penalty_weight * overspeed**2
        drive_reward = w.drive_reward_weight * max(0.0, drive)
        elapsed_time = self.step_count * (self.timestep * self.action_repeat / 1000.0)
        survival_reward = 0.0 if is_stuck else w.survival_reward_weight * elapsed_time

        pursuer_terms = [
            line_of_sight_reward,
        ]
        if w.include_moving_away_reward:
            pursuer_terms.append(moving_away_reward)
        if w.include_visual_moving_away_reward:
            pursuer_terms.append(visual_moving_away_reward)

        obstacle_terms = [
            front_obstacle_penalty,
            side_obstacle_penalty,
            back_obstacle_penalty,
            obstacle_clearance_delta_reward,
            back_approach_penalty,
            obstacle_stall_penalty,
            turn_towards_obstacle_penalty,
            path_prediction_reward,
        ]
        movement_terms = [
            movement_reward,
            still_penalty,
            drive_reward,
            exploration_reward,
        ]
        stability_terms = [
            fast_turn_penalty,
            tight_turn_penalty,
            accelerating_turn_penalty,
            clear_front_turn_penalty,
            straighten_reward,
            action_smoothness_penalty,
            tilt_penalty,
            overspeed_penalty,
        ]
        if w.include_steering_penalty:
            stability_terms.append(steering_penalty)
        survival_terms = [
            survival_reward,
        ]

        reward_groups = {
            "pursuer": pursuer_terms,
            "obstacle": obstacle_terms,
            "movement": movement_terms,
            "stability": stability_terms,
            "survival": survival_terms,
        }

        reward_parts = {
            "obstacle_penalty": 0.0,
            "pursuer_reward_total": float(sum(pursuer_terms)),
            "obstacle_reward_total": float(sum(obstacle_terms)),
            "movement_reward_total": float(sum(movement_terms)),
            "stability_reward_total": float(sum(stability_terms)),
            "survival_reward_total": float(sum(survival_terms)),
            "line_of_sight_reward": float(line_of_sight_reward),
            "visual_moving_away_reward": float(visual_moving_away_reward),
            "exploration_reward": float(exploration_reward),
            "pursuer_visible": float(vision["visible"]),
            "front_pursuer_visible": float(vision["front_visible"]),
            "back_pursuer_visible": float(vision["back_visible"]),
            "pursuer_visual_size": float(vision["visual_size"]),
            "front_obstacle_penalty": float(front_obstacle_penalty),
            "side_obstacle_penalty": float(side_obstacle_penalty),
            "back_obstacle_penalty": float(back_obstacle_penalty),
            "obstacle_clearance_delta_reward": float(obstacle_clearance_delta_reward),
            "back_approach_penalty": float(back_approach_penalty),
            "moving_away_reward": float(moving_away_reward),
            "movement_reward": float(movement_reward),
            "still_penalty": float(still_penalty),
            "obstacle_stall_penalty": float(obstacle_stall_penalty),
            "steering_penalty": float(steering_penalty),
            "fast_turn_penalty": float(fast_turn_penalty),
            "tight_turn_penalty": float(tight_turn_penalty),
            "accelerating_turn_penalty": float(accelerating_turn_penalty),
            "clear_front_turn_penalty": float(clear_front_turn_penalty),
            "straighten_reward": float(straighten_reward),
            "turn_towards_obstacle_penalty": float(turn_towards_obstacle_penalty),
            "path_prediction_reward": float(path_prediction_reward),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "tilt_penalty": float(tilt_penalty),
            "overspeed_penalty": float(overspeed_penalty),
            "survival_reward": float(survival_reward),
        }
        self.previous_pursuer_visible = bool(vision["visible"])
        self.previous_pursuer_visual_size = float(vision["visual_size"])
        return float(sum(sum(terms) for terms in reward_groups.values())), reward_parts

    def _line_of_sight_reward(self, vision: dict[str, float]) -> float:
        visible = bool(vision["visible"])
        if visible:
            return self.reward_weights.line_of_sight_visible_penalty

        reward = self.reward_weights.line_of_sight_hidden_reward
        if self.previous_pursuer_visible:
            reward += self.reward_weights.line_of_sight_break_bonus
        return reward

    def _visual_moving_away_reward(self, vision: dict[str, float]) -> float:
        current_size = float(vision["visual_size"])
        if current_size <= 0.0 or self.previous_pursuer_visual_size <= 0.0:
            return 0.0
        # A smaller recognition footprint is a camera-local proxy for increasing distance.
        return self.reward_weights.visual_moving_away_weight * (self.previous_pursuer_visual_size - current_size)

    def _turn_towards_obstacle_penalty(self) -> float:
        front_ranges = self.directional_lidar_ranges.get("front", np.array([], dtype=np.float32))
        front_ranges = self._filter_lidar_ranges(front_ranges)
        if front_ranges.size < 2 or abs(self.current_steering) < 1e-6:
            return 0.0

        midpoint = front_ranges.size // 2
        left_danger = float(np.mean(np.exp(-front_ranges[:midpoint] / 5.0)))
        right_danger = float(np.mean(np.exp(-front_ranges[midpoint:] / 5.0)))
        danger_delta = left_danger - right_danger

        steering_towards_danger = (
            (self.current_steering < 0.0 and danger_delta > 0.0)
            or (self.current_steering > 0.0 and danger_delta < 0.0)
        )
        if not steering_towards_danger:
            return 0.0
        return self.reward_weights.turn_towards_obstacle_penalty_weight * abs(self.current_steering) * abs(danger_delta)

    def _path_prediction_reward(self, speed_kmh: float, sector_distances: dict[str, float]) -> float:
        speed_mps = max(0.0, speed_kmh / 3.6)
        if speed_mps < 0.2:
            return 0.0

        front_ranges = self.directional_lidar_ranges.get("front", np.array([], dtype=np.float32))
        front_ranges = self._filter_lidar_ranges(front_ranges)
        steering_fraction = float(np.clip(self.current_steering / 0.55, -1.0, 1.0))
        speed_fraction = float(np.clip(speed_mps / max(2.0 * self.pursuer_speed_mps, 1.0), 0.0, 1.0))

        front_risk = 0.0
        if front_ranges.size > 0:
            ray_offsets = np.linspace(-1.0, 1.0, front_ranges.size, dtype=np.float32)
            corridor_width = 0.22 + 0.18 * speed_fraction
            path_weights = np.exp(-((ray_offsets - steering_fraction) ** 2) / (2.0 * corridor_width**2))
            predicted_travel = speed_mps * 1.4 + 2.5
            closing_risk = np.clip((predicted_travel - front_ranges) / predicted_travel, 0.0, 1.0)
            front_risk = float(np.max(path_weights * closing_risk))

        turn_strength = abs(steering_fraction) * speed_fraction
        turn_side_distance = sector_distances["right"] if steering_fraction > 0.0 else sector_distances["left"]
        side_risk = turn_strength * max(0.0, (3.5 - turn_side_distance) / 3.5)
        path_risk = max(front_risk, side_risk)

        clear_path_reward = self.reward_weights.clear_path_reward_weight * speed_fraction * (1.0 - path_risk)
        predicted_collision_penalty = self.reward_weights.predicted_collision_penalty_weight * path_risk
        return float(clear_path_reward + predicted_collision_penalty)

    def _front_obstacle_penalty(self) -> float:
        front_ranges = self.directional_lidar_ranges.get("front", np.array([], dtype=np.float32))
        front_ranges = self._filter_lidar_ranges(front_ranges)
        if front_ranges.size == 0:
            return 0.0

        center_offsets = np.linspace(-1.0, 1.0, front_ranges.size, dtype=np.float32)
        center_weights = 1.0 - np.abs(center_offsets)
        center_weights = 0.25 + 0.75 * center_weights
        danger = center_weights * np.exp(-front_ranges / 8.0)
        return self.reward_weights.front_obstacle_penalty_weight * float(np.max(danger))

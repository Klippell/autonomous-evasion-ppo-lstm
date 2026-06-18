"""Reward shaping for the evader Gym environment."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Typed reward configuration shared by JSON configs and the environment."""

    # Pursuer / camera rewards.
    moving_away_positive_weight: float = 4.0
    moving_away_negative_weight: float = 8.0
    pursuer_distance_delta_limit: float = 2.0
    radial_escape_movement_limit: float = 2.0
    include_moving_away_reward: bool = False
    line_of_sight_hidden_reward: float = 1.0
    line_of_sight_break_bonus: float = 5.0
    line_of_sight_visible_penalty: float = -0.5
    visual_moving_away_weight: float = 4.0
    include_visual_moving_away_reward: bool = True
    exploration_new_cell_reward: float = 1.5
    exploration_revisit_penalty: float = -0.05
    radial_escape_reward_weight: float = 4.0
    tangential_orbit_penalty_weight: float = -4.0
    tangential_orbit_penalty_distance: float = 30.0

    # Obstacle rewards.
    front_obstacle_penalty_weight: float = -4.0
    side_obstacle_penalty_weight: float = -0.7
    back_obstacle_penalty_weight: float = -0.3
    front_clearance_delta_weight: float = 2.0
    side_clearance_delta_weight: float = 0.8
    back_clearance_delta_weight: float = 0.4
    clearance_delta_limit: float = 2.0
    back_approach_penalty_weight: float = -1.0
    obstacle_stall_penalty: float = -0.5
    obstacle_collision_free_reward: float = 0.15
    obstacle_approach_penalty_scale: float = 0.15
    front_blocked_movement_scale: float = 0.15
    front_blocked_straight_penalty_weight: float = -8.0
    obstacle_risk_escape_reward_min_scale: float = 0.0
    obstacle_action_reward_weight: float = 4.0
    obstacle_action_penalty_weight: float = -14.0
    obstacle_action_risk_threshold: float = 0.05
    obstacle_action_required_steering_min: float = 0.25
    obstacle_turn_commitment_reward_weight: float = 4.0
    obstacle_turn_switch_penalty_weight: float = -10.0
    obstacle_turn_direction_signal_threshold: float = 0.18
    obstacle_turn_release_risk: float = 0.08
    obstacle_turn_progress_reward_weight: float = 3.0
    obstacle_wrong_heading_penalty_weight: float = -3.0
    obstacle_insufficient_turn_penalty_weight: float = -3.0
    obstacle_safety_intervention_penalty_weight: float = -8.0

    # Movement / progress rewards.
    still_penalty: float = -3.0
    not_stuck_reward: float = 0.2
    movement_reward_weight: float = 1.5
    movement_distance_reward_limit: float = 2.0
    drive_reward_weight: float = 0.08
    survival_reward_weight: float = 0.02
    survival_reward_growth_weight: float = 0.0
    survival_reward_cap: float = 0.2

    # Stability / control rewards.
    steering_penalty_weight: float = -0.01
    include_steering_penalty: bool = False
    fast_turn_penalty_weight: float = -0.6
    tight_turn_penalty_weight: float = -1.0
    long_curve_penalty_weight: float = -0.4
    long_curve_free_steps: int = 8
    straight_drive_reward_weight: float = 0.8
    clear_path_straight_reward_weight: float = 1.0
    unnecessary_turn_penalty_weight: float = -1.0
    clear_path_reward_risk_threshold: float = 0.05
    avoidance_turn_reward_weight: float = 1.2
    over_avoidance_turn_penalty_weight: float = -2.0
    front_obstacle_avoidance_distance: float = 14.0
    clear_front_turn_penalty_weight: float = -0.8
    straighten_reward_weight: float = 1.0
    turn_towards_obstacle_penalty_weight: float = -1.0
    turn_towards_visible_pursuer_penalty_weight: float = -4.0
    clear_path_reward_weight: float = 0.8
    predicted_collision_penalty_weight: float = -3.0
    action_smoothness_penalty_weight: float = -0.15
    tilt_penalty_weight: float = -5.0
    overspeed_penalty_weight: float = -4.0


def reward_weights_from_mapping(values: dict[str, object] | RewardWeights | None) -> RewardWeights:
    """Validate a config mapping before constructing immutable reward weights."""

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
    """Reward calculation separated from Webots lifecycle and device handling."""

    def _reward(
        self,
        distance: float,
        sector_distances: dict[str, float],
        action: np.ndarray,
        speed_kmh: float,
        moved_distance: float,
        evader_xy: np.ndarray,
        pursuer_xy: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Calculate the scalar reward and expose every component for diagnostics."""

        w = self.reward_weights
        vision = self._camera_pursuer_observation()
        drive = float(action[1])
        front_corridor_risk, front_left_risk, front_right_risk = self._front_lidar_risk(
            self.reward_weights.front_obstacle_avoidance_distance
        )
        clear_path_scale = self._clear_path_scale(front_corridor_risk)
        raw_distance_delta = distance - self.previous_distance
        distance_delta = float(
            np.clip(
                raw_distance_delta,
                -w.pursuer_distance_delta_limit,
                w.pursuer_distance_delta_limit,
            )
        )
        if distance_delta > 0:
            moving_away_reward = w.moving_away_positive_weight * distance_delta
        else:
            moving_away_reward = w.moving_away_negative_weight * distance_delta
        line_of_sight_reward = self._line_of_sight_reward(vision)
        visual_moving_away_reward = self._visual_moving_away_reward(vision)
        radial_escape_reward, tangential_orbit_penalty = self._radial_escape_reward(evader_xy, pursuer_xy, distance)
        # Positive escape rewards fade near a frontal obstacle. Penalties remain
        # intact so driving into danger cannot be profitable overall.
        obstacle_risk_escape_scale = 1.0 - (
            1.0 - w.obstacle_risk_escape_reward_min_scale
        ) * front_corridor_risk
        moving_away_reward = self._scale_positive_reward(moving_away_reward, obstacle_risk_escape_scale)
        line_of_sight_reward = self._scale_positive_reward(line_of_sight_reward, obstacle_risk_escape_scale)
        visual_moving_away_reward = self._scale_positive_reward(visual_moving_away_reward, obstacle_risk_escape_scale)
        radial_escape_reward = self._scale_positive_reward(radial_escape_reward, obstacle_risk_escape_scale)
        exploration_reward = self._exploration_reward()
        front_obstacle_penalty = self._front_obstacle_penalty()
        side_obstacle_penalty = w.side_obstacle_penalty_weight * (
            math.exp(-sector_distances["left"] / 3.0)
            + math.exp(-sector_distances["right"] / 3.0)
        )
        back_obstacle_penalty = w.back_obstacle_penalty_weight * math.exp(-sector_distances["back"] / 2.5)
        obstacle_clearance_delta_reward = (
            self._clearance_delta_reward("front", sector_distances["front"], w.front_clearance_delta_weight)
            + self._clearance_delta_reward("left", sector_distances["left"], w.side_clearance_delta_weight)
            + self._clearance_delta_reward("right", sector_distances["right"], w.side_clearance_delta_weight)
            + self._clearance_delta_reward("back", sector_distances["back"], w.back_clearance_delta_weight)
        )
        back_distance = sector_distances["back"]
        back_approach = max(0.0, self.previous_sector_distances["back"] - back_distance)
        back_danger_range = 8.0
        back_proximity = max(0.0, (back_danger_range - back_distance) / back_danger_range)
        back_approach_penalty = w.back_approach_penalty_weight * back_approach * (1.0 + 4.0 * back_proximity)
        is_stuck = moved_distance < self.still_distance_threshold
        still_penalty = w.still_penalty if is_stuck else w.not_stuck_reward
        movement_scale = 1.0 - (1.0 - w.front_blocked_movement_scale) * front_corridor_risk
        rewarded_moved_distance = min(moved_distance, w.movement_distance_reward_limit)
        movement_reward = w.movement_reward_weight * rewarded_moved_distance * movement_scale
        obstacle_stall_penalty = (
            w.obstacle_stall_penalty
            if moved_distance < self.still_distance_threshold and sector_distances["front"] < 8.0
            else 0.0
        )
        obstacle_collision_free_reward = self._obstacle_collision_free_reward(sector_distances)
        steering_penalty = w.steering_penalty_weight * abs(self.current_steering)
        normalized_speed = np.clip(speed_kmh / 70.0, 0.0, 1.0)
        fast_turn_penalty = w.fast_turn_penalty_weight * normalized_speed * abs(self.current_steering)
        steering_fraction = abs(float(np.clip(self.current_steering / 0.55, -1.0, 1.0)))
        tight_turn_penalty = w.tight_turn_penalty_weight * max(0.0, steering_fraction - 0.55) ** 2
        long_curve_steps = max(0, self.steering_direction_streak - w.long_curve_free_steps)
        long_curve_penalty = (
            w.long_curve_penalty_weight
            * long_curve_steps
            * steering_fraction
            * normalized_speed
            * (1.0 - front_corridor_risk) ** 2
        )
        straight_drive_reward = (
            w.straight_drive_reward_weight
            * max(0.0, drive)
            * (1.0 - steering_fraction)
            * clear_path_scale
        )
        (
            clear_path_straight_reward,
            unnecessary_turn_penalty,
            avoidance_turn_reward,
            over_avoidance_turn_penalty,
        ) = self._obstacle_avoidance_steering_rewards(steering_fraction, drive)
        (
            obstacle_action_reward,
            obstacle_action_penalty,
            obstacle_target_steering,
            obstacle_target_drive,
            obstacle_steering_error,
            obstacle_drive_error,
            obstacle_turn_commitment_reward,
            obstacle_turn_switch_penalty,
        ) = self._obstacle_action_target_reward(front_corridor_risk, front_left_risk, front_right_risk, drive)
        (
            obstacle_turn_progress_reward,
            obstacle_wrong_heading_penalty,
            obstacle_insufficient_turn_penalty,
            obstacle_heading_progress,
            obstacle_heading_goal,
        ) = self._obstacle_heading_progress_rewards(front_corridor_risk, front_left_risk, front_right_risk, drive, speed_kmh)
        front_clearance = sector_distances["front"]
        front_clear_fraction = float(np.clip((front_clearance - 8.0) / 10.0, 0.0, 1.0)) * clear_path_scale
        previous_steering_fraction = abs(float(np.clip(self.previous_action[0] / 0.55, -1.0, 1.0)))
        clear_front_turn_penalty = w.clear_front_turn_penalty_weight * steering_fraction * front_clear_fraction
        straighten_reward = (
            w.straighten_reward_weight
            * max(0.0, previous_steering_fraction - steering_fraction)
            * front_clear_fraction
        )
        turn_towards_obstacle_penalty = self._turn_towards_obstacle_penalty()
        turn_towards_visible_pursuer_penalty = self._turn_towards_visible_pursuer_penalty(vision, steering_fraction)
        path_prediction_reward = self._path_prediction_reward(speed_kmh, sector_distances)
        action_smoothness_penalty = w.action_smoothness_penalty_weight * float(np.linalg.norm(action - self.previous_action))
        tilt_penalty = w.tilt_penalty_weight * max(0.0, self._evader_tilt_angle() - 0.35)
        speed_mps = speed_kmh / 3.6
        overspeed = max(0.0, speed_mps - self._evader_speed_limit_mps())
        overspeed_penalty = w.overspeed_penalty_weight * overspeed**2
        required_turn_fraction = min(1.0, abs(w.obstacle_action_required_steering_min) / 0.55)
        missing_turn_fraction = max(0.0, required_turn_fraction - steering_fraction) / max(required_turn_fraction, 1e-6)
        front_blocked_straight_penalty = (
            w.front_blocked_straight_penalty_weight
            * max(0.0, 1.0 - clear_path_scale)
            * missing_turn_fraction
            * (0.4 + 0.6 * max(0.0, drive))
        )
        obstacle_safety_intervention_penalty = (
            w.obstacle_safety_intervention_penalty_weight
            * float(getattr(self, "obstacle_safety_action_delta", 0.0))
        )
        drive_reward = w.drive_reward_weight * max(0.0, drive) * clear_path_scale
        elapsed_time = self.step_count * (self.timestep * self.action_repeat / 1000.0)
        survival_reward = 0.0
        if not is_stuck:
            survival_reward = min(
                w.survival_reward_weight + w.survival_reward_growth_weight * elapsed_time,
                w.survival_reward_cap,
            )

        # Group totals are logged independently in TensorBoard and debug CSVs.
        # Keeping each component in exactly one group avoids double counting.
        pursuer_terms = [
            line_of_sight_reward,
            radial_escape_reward,
            tangential_orbit_penalty,
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
            obstacle_collision_free_reward,
            back_approach_penalty,
            obstacle_stall_penalty,
            front_blocked_straight_penalty,
            obstacle_action_reward,
            obstacle_action_penalty,
            obstacle_turn_commitment_reward,
            obstacle_turn_switch_penalty,
            obstacle_turn_progress_reward,
            obstacle_wrong_heading_penalty,
            obstacle_insufficient_turn_penalty,
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
            long_curve_penalty,
            straight_drive_reward,
            clear_path_straight_reward,
            unnecessary_turn_penalty,
            avoidance_turn_reward,
            over_avoidance_turn_penalty,
            clear_front_turn_penalty,
            straighten_reward,
            action_smoothness_penalty,
            tilt_penalty,
            overspeed_penalty,
            turn_towards_visible_pursuer_penalty,
            obstacle_safety_intervention_penalty,
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
            "radial_escape_reward": float(radial_escape_reward),
            "tangential_orbit_penalty": float(tangential_orbit_penalty),
            "raw_distance_delta": float(raw_distance_delta),
            "distance_delta": float(distance_delta),
            "exploration_reward": float(exploration_reward),
            "pursuer_visible": float(vision["visible"]),
            "front_pursuer_visible": float(vision["front_visible"]),
            "back_pursuer_visible": float(vision["back_visible"]),
            "left_pursuer_visible": float(vision["left_visible"]),
            "right_pursuer_visible": float(vision["right_visible"]),
            "left_pursuer_visual_size": float(vision["left_size"]),
            "right_pursuer_visual_size": float(vision["right_size"]),
            "pursuer_visual_size": float(vision["visual_size"]),
            "front_obstacle_penalty": float(front_obstacle_penalty),
            "side_obstacle_penalty": float(side_obstacle_penalty),
            "back_obstacle_penalty": float(back_obstacle_penalty),
            "obstacle_clearance_delta_reward": float(obstacle_clearance_delta_reward),
            "obstacle_collision_free_reward": float(obstacle_collision_free_reward),
            "back_approach_penalty": float(back_approach_penalty),
            "front_corridor_risk": float(front_corridor_risk),
            "obstacle_risk_escape_scale": float(obstacle_risk_escape_scale),
            "front_blocked_straight_penalty": float(front_blocked_straight_penalty),
            "obstacle_action_reward": float(obstacle_action_reward),
            "obstacle_action_penalty": float(obstacle_action_penalty),
            "obstacle_target_steering": float(obstacle_target_steering),
            "obstacle_target_drive": float(obstacle_target_drive),
            "obstacle_steering_error": float(obstacle_steering_error),
            "obstacle_drive_error": float(obstacle_drive_error),
            "obstacle_turn_commitment_reward": float(obstacle_turn_commitment_reward),
            "obstacle_turn_switch_penalty": float(obstacle_turn_switch_penalty),
            "obstacle_turn_progress_reward": float(obstacle_turn_progress_reward),
            "obstacle_wrong_heading_penalty": float(obstacle_wrong_heading_penalty),
            "obstacle_insufficient_turn_penalty": float(obstacle_insufficient_turn_penalty),
            "obstacle_heading_progress": float(obstacle_heading_progress),
            "obstacle_heading_goal": float(obstacle_heading_goal),
            "obstacle_safety_intervention_penalty": float(obstacle_safety_intervention_penalty),
            "moving_away_reward": float(moving_away_reward),
            "movement_reward": float(movement_reward),
            "rewarded_moved_distance": float(rewarded_moved_distance),
            "still_penalty": float(still_penalty),
            "obstacle_stall_penalty": float(obstacle_stall_penalty),
            "steering_penalty": float(steering_penalty),
            "fast_turn_penalty": float(fast_turn_penalty),
            "tight_turn_penalty": float(tight_turn_penalty),
            "long_curve_penalty": float(long_curve_penalty),
            "straight_drive_reward": float(straight_drive_reward),
            "clear_path_straight_reward": float(clear_path_straight_reward),
            "unnecessary_turn_penalty": float(unnecessary_turn_penalty),
            "avoidance_turn_reward": float(avoidance_turn_reward),
            "over_avoidance_turn_penalty": float(over_avoidance_turn_penalty),
            "clear_front_turn_penalty": float(clear_front_turn_penalty),
            "straighten_reward": float(straighten_reward),
            "turn_towards_obstacle_penalty": float(turn_towards_obstacle_penalty),
            "turn_towards_visible_pursuer_penalty": float(turn_towards_visible_pursuer_penalty),
            "path_prediction_reward": float(path_prediction_reward),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "tilt_penalty": float(tilt_penalty),
            "overspeed_penalty": float(overspeed_penalty),
            "survival_reward": float(survival_reward),
        }
        self.previous_pursuer_visible = bool(vision["visible"])
        self.previous_pursuer_visual_size = float(vision["visual_size"])
        return float(sum(sum(terms) for terms in reward_groups.values())), reward_parts

    @staticmethod
    def _scale_positive_reward(value: float, scale: float) -> float:
        return value * scale if value > 0.0 else value

    def _clear_path_scale(self, front_risk: float) -> float:
        threshold = max(self.reward_weights.clear_path_reward_risk_threshold, 1e-6)
        return float(np.clip(1.0 - front_risk / threshold, 0.0, 1.0))

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

    def _radial_escape_reward(
        self,
        evader_xy: np.ndarray,
        pursuer_xy: np.ndarray,
        distance: float,
    ) -> tuple[float, float]:
        if self.previous_position is None:
            return 0.0, 0.0
        if not np.all(np.isfinite(evader_xy)) or not np.all(np.isfinite(pursuer_xy)) or not np.all(np.isfinite(self.previous_position)):
            return 0.0, 0.0
        if distance <= 1e-6:
            return 0.0, 0.0

        displacement = evader_xy - self.previous_position
        away_direction = (evader_xy - pursuer_xy) / distance
        radial_movement = float(
            np.clip(
                np.dot(displacement, away_direction),
                -self.reward_weights.radial_escape_movement_limit,
                self.reward_weights.radial_escape_movement_limit,
            )
        )
        tangential_vector = displacement - radial_movement * away_direction
        tangential_movement = min(
            float(np.linalg.norm(tangential_vector)),
            self.reward_weights.radial_escape_movement_limit,
        )

        radial_reward = self.reward_weights.radial_escape_reward_weight * radial_movement
        close_fraction = max(0.0, 1.0 - distance / self.reward_weights.tangential_orbit_penalty_distance)
        orbit_penalty = self.reward_weights.tangential_orbit_penalty_weight * tangential_movement * close_fraction
        return radial_reward, orbit_penalty

    def _turn_towards_visible_pursuer_penalty(self, vision: dict[str, float], steering_fraction: float) -> float:
        if steering_fraction <= 1e-6:
            return 0.0

        turning_towards_left = self.current_steering < 0.0 and bool(vision["left_visible"])
        turning_towards_right = self.current_steering > 0.0 and bool(vision["right_visible"])
        if not turning_towards_left and not turning_towards_right:
            return 0.0

        visible_size = max(float(vision["left_size"]), float(vision["right_size"]))
        visibility_scale = 1.0 + 2.0 * visible_size
        return self.reward_weights.turn_towards_visible_pursuer_penalty_weight * steering_fraction * visibility_scale

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

    def _obstacle_avoidance_steering_rewards(self, steering_fraction: float, drive: float) -> tuple[float, float, float, float]:
        drive_fraction = max(0.0, drive)
        if drive_fraction <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        front_risk, left_danger, right_danger = self._front_lidar_risk(
            self.reward_weights.front_obstacle_avoidance_distance
        )

        clear_fraction = self._clear_path_scale(front_risk)
        clear_path_straight_reward = (
            self.reward_weights.clear_path_straight_reward_weight
            * drive_fraction
            * clear_fraction
            * (1.0 - steering_fraction)
        )
        unnecessary_turn_penalty = (
            self.reward_weights.unnecessary_turn_penalty_weight
            * drive_fraction
            * clear_fraction
            * steering_fraction
        )

        required_steering = front_risk * 0.75
        if front_risk > 0.15:
            required_steering = max(required_steering, 0.25)
        over_steering = max(0.0, steering_fraction - required_steering)
        over_avoidance_turn_penalty = (
            self.reward_weights.over_avoidance_turn_penalty_weight
            * drive_fraction
            * over_steering**2
        )

        danger_delta = left_danger - right_danger
        has_directional_signal = abs(danger_delta) > 0.03
        if self.current_steering > 0.0:
            turning_away = danger_delta > 0.0
        elif self.current_steering < 0.0:
            turning_away = danger_delta < 0.0
        else:
            turning_away = False

        if has_directional_signal:
            alignment = 1.0 if turning_away else -0.5
        else:
            alignment = 1.0 if steering_fraction > 0.0 else 0.0

        useful_steering = min(steering_fraction, required_steering)
        avoidance_turn_reward = (
            self.reward_weights.avoidance_turn_reward_weight
            * drive_fraction
            * front_risk
            * useful_steering
            * alignment
        )
        return clear_path_straight_reward, unnecessary_turn_penalty, avoidance_turn_reward, over_avoidance_turn_penalty

    def _obstacle_action_target_reward(
        self,
        front_risk: float,
        left_risk: float,
        right_risk: float,
        drive: float,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        threshold = float(np.clip(self.reward_weights.obstacle_action_risk_threshold, 0.0, 0.95))
        active = float(np.clip((front_risk - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        if active <= 0.0:
            if front_risk <= self.reward_weights.obstacle_turn_release_risk:
                self.obstacle_turn_direction = 0
            return 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0

        min_steering = abs(self.reward_weights.obstacle_action_required_steering_min)
        required_abs_steering = min(0.50, max(min_steering, 0.20 + 0.35 * front_risk))
        danger_delta = left_risk - right_risk
        current_sign = self._sign(self.current_steering)
        previous_sign = self._sign(float(self.previous_action[0]))
        committed_sign = int(getattr(self, "obstacle_turn_direction", 0))

        if abs(danger_delta) > self.reward_weights.obstacle_turn_direction_signal_threshold:
            committed_sign = 1 if danger_delta > 0.0 else -1
        elif committed_sign == 0:
            committed_sign = previous_sign or current_sign or 1

        self.obstacle_turn_direction = committed_sign
        target_steering = committed_sign * required_abs_steering
        steering_error = abs(self.current_steering - target_steering) / 0.55

        target_drive = 0.0
        drive_error = max(0.0, -drive) / 1.5
        steering_error = float(np.clip(steering_error, 0.0, 1.5))
        drive_error = float(np.clip(drive_error, 0.0, 1.5))

        penalty = (
            self.reward_weights.obstacle_action_penalty_weight
            * active
            * (0.40 * drive_error + 0.60 * steering_error)
        )
        good_action = max(0.0, 1.0 - drive_error) * max(0.0, 1.0 - steering_error)
        reward = self.reward_weights.obstacle_action_reward_weight * active * good_action
        aligned = current_sign == committed_sign and abs(self.current_steering) >= min_steering - 1e-6
        switched = current_sign != 0 and current_sign != committed_sign
        commitment_reward = (
            self.reward_weights.obstacle_turn_commitment_reward_weight
            * active
            * float(aligned)
            * max(0.0, 1.0 - drive_error)
        )
        switch_penalty = self.reward_weights.obstacle_turn_switch_penalty_weight * active * float(switched)
        return reward, penalty, target_steering, target_drive, steering_error, drive_error, commitment_reward, switch_penalty

    def _obstacle_heading_progress_rewards(
        self,
        front_risk: float,
        left_risk: float,
        right_risk: float,
        drive: float,
        speed_kmh: float,
    ) -> tuple[float, float, float, float, float]:
        threshold = float(np.clip(self.reward_weights.obstacle_action_risk_threshold, 0.0, 0.95))
        active = float(np.clip((front_risk - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        if active <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, self._required_obstacle_heading_change(front_risk)

        progress = self._obstacle_heading_progress(front_risk, left_risk, right_risk)
        goal = max(float(progress["goal"]), 1e-6)
        signed_progress = float(progress["signed"])
        progress_fraction = float(np.clip(max(0.0, signed_progress) / goal, 0.0, 1.0))
        wrong_fraction = float(np.clip(max(0.0, -signed_progress) / goal, 0.0, 1.0))
        remaining_fraction = float(np.clip((goal - max(0.0, signed_progress)) / goal, 0.0, 1.0))

        speed_scale = 0.25 + 0.75 * float(np.clip((speed_kmh / 3.6) / 6.0, 0.0, 1.0))
        forward_drive = max(0.0, drive)
        obstacle_turn_progress_reward = (
            self.reward_weights.obstacle_turn_progress_reward_weight
            * active
            * progress_fraction
            * speed_scale
        )
        obstacle_wrong_heading_penalty = (
            self.reward_weights.obstacle_wrong_heading_penalty_weight
            * active
            * wrong_fraction
            * (0.5 + 0.5 * speed_scale)
        )
        obstacle_insufficient_turn_penalty = (
            self.reward_weights.obstacle_insufficient_turn_penalty_weight
            * active
            * remaining_fraction
            * forward_drive
            * (0.35 + 0.65 * front_risk)
        )
        return (
            float(obstacle_turn_progress_reward),
            float(obstacle_wrong_heading_penalty),
            float(obstacle_insufficient_turn_penalty),
            signed_progress,
            goal,
        )

    @staticmethod
    def _sign(value: float) -> int:
        if value > 1e-6:
            return 1
        if value < -1e-6:
            return -1
        return 0

    def _front_lidar_risk(self, danger_range: float, center_sigma: float = 0.34) -> tuple[float, float, float]:
        front_ranges = self.directional_lidar_ranges.get("front", np.array([], dtype=np.float32))
        front_ranges = self._filter_lidar_ranges(front_ranges)
        if front_ranges.size == 0:
            return 0.0, 0.0, 0.0

        ray_offsets = np.linspace(-1.0, 1.0, front_ranges.size, dtype=np.float32)
        center_weights = np.exp(-(ray_offsets**2) / (2.0 * center_sigma**2))
        danger_range = max(danger_range, 1.0)
        ray_risk = np.clip((danger_range - front_ranges) / danger_range, 0.0, 1.0)
        front_risk = float(np.max(center_weights * ray_risk))

        midpoint = front_ranges.size // 2
        left_risk = float(np.mean(ray_risk[:midpoint])) if midpoint > 0 else 0.0
        right_risk = float(np.mean(ray_risk[midpoint:])) if midpoint < front_ranges.size else 0.0
        return front_risk, left_risk, right_risk

    def _clearance_delta_reward(self, direction: str, current_distance: float, weight: float) -> float:
        delta = float(
            np.clip(
                current_distance - self.previous_sector_distances[direction],
                -self.reward_weights.clearance_delta_limit,
                self.reward_weights.clearance_delta_limit,
            )
        )
        if delta >= 0.0:
            return weight * delta
        return self.reward_weights.obstacle_approach_penalty_scale * abs(weight) * delta

    def _obstacle_collision_free_reward(self, sector_distances: dict[str, float]) -> float:
        min_distance = min(sector_distances.values())
        collision_distance = self._lidar_max_range() * 0.015
        if min_distance <= collision_distance:
            return 0.0
        return self.reward_weights.obstacle_collision_free_reward

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
        danger = center_weights * np.clip((8.0 - front_ranges) / 8.0, 0.0, 1.0)
        return self.reward_weights.front_obstacle_penalty_weight * float(np.max(danger))

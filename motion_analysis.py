from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from utils.math_utils import angle_between_vectors

logger = logging.getLogger(__name__)


@dataclass
class JointMetrics:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass
class MotionReport:
    angles_deg: Dict[str, float] = field(default_factory=dict)
    segment_lengths: Dict[str, float] = field(default_factory=dict)
    joint_metrics: Dict[str, JointMetrics] = field(default_factory=dict)
    math_operations: List[str] = field(default_factory=list)


class MotionAnalyzer:
    """Compute motion vectors, angles, lengths, velocity, and acceleration."""

    def __init__(self) -> None:
        self._previous_positions: Dict[str, np.ndarray] = {}
        self._previous_velocities: Dict[str, np.ndarray] = {}

    def analyze(self, points_3d: Dict[str, np.ndarray], delta_t: float) -> MotionReport:
        report = MotionReport()

        # Segment lengths (Euclidean distances)
        segments = {
            "left_upper_arm": ("left_shoulder", "left_elbow"),
            "left_lower_arm": ("left_elbow", "left_wrist"),
            "right_upper_arm": ("right_shoulder", "right_elbow"),
            "right_lower_arm": ("right_elbow", "right_wrist"),
            "left_upper_leg": ("left_hip", "left_knee"),
            "left_lower_leg": ("left_knee", "left_ankle"),
            "right_upper_leg": ("right_hip", "right_knee"),
            "right_lower_leg": ("right_knee", "right_ankle"),
        }
        for name, (a, b) in segments.items():
            if a in points_3d and b in points_3d:
                vector = points_3d[b] - points_3d[a]
                length = float(np.linalg.norm(vector))
                report.segment_lengths[name] = length
                report.math_operations.append(
                    f"Segment {name}: length = ||{b} - {a}|| = sqrt(sum((x_i^b - x_i^a)^2))"
                )

        # Angles at elbows and knees
        angle_pairs = {
            "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
            "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
            "left_knee": ("left_hip", "left_knee", "left_ankle"),
            "right_knee": ("right_hip", "right_knee", "right_ankle"),
        }
        for name, (a, b, c) in angle_pairs.items():
            if a in points_3d and b in points_3d and c in points_3d:
                v1 = points_3d[a] - points_3d[b]
                v2 = points_3d[c] - points_3d[b]
                angle = angle_between_vectors(v1, v2)
                report.angles_deg[name] = angle
                report.math_operations.append(
                    f"Angle {name}: arccos( (v1·v2) / (||v1||*||v2||) )"
                )

        # Velocity and acceleration
        for name, position in points_3d.items():
            prev_pos = self._previous_positions.get(name)
            velocity = np.zeros_like(position)
            acceleration = np.zeros_like(position)
            if prev_pos is not None and delta_t > 0:
                velocity = (position - prev_pos) / delta_t
                prev_vel = self._previous_velocities.get(name)
                if prev_vel is not None:
                    acceleration = (velocity - prev_vel) / delta_t
                report.math_operations.append(
                    f"Velocity {name}: v = (p_t - p_(t-1)) / Δt; Accel a = (v_t - v_(t-1)) / Δt"
                )

            report.joint_metrics[name] = JointMetrics(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
            )
            self._previous_positions[name] = position
            self._previous_velocities[name] = velocity

        return report

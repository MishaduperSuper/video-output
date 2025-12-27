from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from utils.math_utils import moving_average
from pose_detector import Keypoint

logger = logging.getLogger(__name__)

SKELETON_CONNECTIONS = [
    ("head", "left_shoulder"),
    ("head", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass
class SkeletonFrame:
    keypoints: Dict[str, Keypoint]
    points_2d: Dict[str, np.ndarray]
    points_3d: Dict[str, np.ndarray]


@dataclass
class SkeletonBuilder:
    """Builds 2D and pseudo-3D skeleton from detected keypoints."""

    smoothing_window: int = 5
    history: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def _smooth_point(self, name: str, point: np.ndarray) -> np.ndarray:
        history = self.history.setdefault(name, [])
        history.append(point)
        return moving_average(history, self.smoothing_window)

    def build(self, keypoints: Dict[str, Keypoint]) -> SkeletonFrame:
        points_2d: Dict[str, np.ndarray] = {}
        points_3d: Dict[str, np.ndarray] = {}

        for name, kp in keypoints.items():
            point_2d = np.array([kp.x, kp.y], dtype=float)
            point_3d = np.array([kp.x, kp.y, kp.z], dtype=float)

            # Interpolation: if visibility is low, reuse last valid position.
            if kp.visibility < 0.5 and name in self.history:
                point_2d = self.history[name][-1][:2]
                point_3d = self.history[name][-1]

            smoothed = self._smooth_point(name, point_3d)
            points_2d[name] = smoothed[:2]
            points_3d[name] = smoothed

        return SkeletonFrame(keypoints=keypoints, points_2d=points_2d, points_3d=points_3d)

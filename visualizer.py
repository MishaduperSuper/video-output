from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from motion_analysis import MotionReport
from skeleton_builder import SKELETON_CONNECTIONS

logger = logging.getLogger(__name__)


@dataclass
class Visualizer:
    output_path: Path
    fps: float
    frame_size: Tuple[int, int]

    def __post_init__(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, self.frame_size)
        if not self._writer.isOpened():
            raise ValueError(f"Unable to open output video for writing: {self.output_path}")

    def draw(self, frame: np.ndarray, points_2d: Dict[str, np.ndarray], report: MotionReport) -> None:
        """Draw skeleton on a BGR frame with color indicating joint speed."""

        for joint, metrics in report.joint_metrics.items():
            speed = float(np.linalg.norm(metrics.velocity))
            color = self._speed_color(speed)
            if joint in points_2d:
                x, y = points_2d[joint].astype(int)
                cv2.circle(frame, (x, y), 5, color, thickness=-1)

        for a, b in SKELETON_CONNECTIONS:
            if a in points_2d and b in points_2d:
                pt_a = tuple(points_2d[a].astype(int))
                pt_b = tuple(points_2d[b].astype(int))
                cv2.line(frame, pt_a, pt_b, (0, 255, 255), thickness=2)

        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    @staticmethod
    def _speed_color(speed: float) -> Tuple[int, int, int]:
        """Map speed to BGR color (slow=blue, fast=red)."""

        speed_norm = min(speed / 50.0, 1.0)
        blue = int(255 * (1 - speed_norm))
        red = int(255 * speed_norm)
        return (blue, 0, red)

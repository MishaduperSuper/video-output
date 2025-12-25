from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class FrameData:
    index: int
    timestamp_s: float
    bgr: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray


class VideoLoader:
    """Loads video frames with FPS normalization and color conversion."""

    def __init__(self, path: str, target_fps: float | None = None) -> None:
        self.path = path
        self.target_fps = target_fps
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_step = 1
        if target_fps and target_fps > 0:
            self._frame_step = max(1, round(self._fps / target_fps))
            logger.info("Normalizing FPS from %.2f to %.2f using step %d", self._fps, target_fps, self._frame_step)

    @property
    def meta(self) -> VideoMeta:
        return VideoMeta(
            fps=self._fps,
            frame_count=self._frame_count,
            width=self._width,
            height=self._height,
        )

    def frames(self) -> Generator[FrameData, None, None]:
        """Yield frames with BGR, RGB, and Grayscale representations."""

        idx = 0
        out_idx = 0
        while True:
            ret, frame = self._capture.read()
            if not ret:
                break
            if idx % self._frame_step != 0:
                idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp_s = out_idx / max(self._fps / self._frame_step, 1e-6)
            yield FrameData(index=out_idx, timestamp_s=timestamp_s, bgr=frame, rgb=rgb, gray=gray)
            out_idx += 1
            idx += 1

    def release(self) -> None:
        self._capture.release()

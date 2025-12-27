from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Vector2D:
    """2D vector with basic math helpers.

    This class exists to make the math in the project explicit and readable.
    """

    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between vectors using cosine formula.

    angle = arccos( (v1 · v2) / (||v1|| * ||v2||) )
    """

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def moving_average(values: Iterable[np.ndarray], window: int) -> np.ndarray:
    """Compute moving average over a window for a sequence of vectors."""

    value_list = list(values)
    if not value_list:
        return np.array([])
    window = max(1, window)
    if len(value_list) < window:
        return np.mean(np.stack(value_list), axis=0)
    stacked = np.stack(value_list[-window:])
    return np.mean(stacked, axis=0)

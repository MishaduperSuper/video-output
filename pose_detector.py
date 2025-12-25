from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

POSE_LANDMARKS = {
    "head": mp.solutions.pose.PoseLandmark.NOSE,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseDetector:
    """MediaPipe Pose wrapper for extracting keypoints."""

    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, rgb_frame: np.ndarray, frame_width: int, frame_height: int) -> Dict[str, Keypoint]:
        """Detect pose keypoints for a single RGB frame."""

        results = self._pose.process(rgb_frame)
        if not results.pose_landmarks:
            logger.debug("No pose landmarks detected")
            return {}

        keypoints: Dict[str, Keypoint] = {}
        for name, landmark_id in POSE_LANDMARKS.items():
            landmark = results.pose_landmarks.landmark[landmark_id]
            keypoints[name] = Keypoint(
                name=name,
                x=landmark.x * frame_width,
                y=landmark.y * frame_height,
                z=landmark.z * frame_width,
                visibility=landmark.visibility,
            )
        return keypoints

    def close(self) -> None:
        self._pose.close()

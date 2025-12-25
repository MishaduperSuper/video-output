from __future__ import annotations

import argparse
import logging
from pathlib import Path

from math_report import MathReportWriter
from motion_analysis import MotionAnalyzer, MotionReport
from pose_detector import PoseDetector
from skeleton_builder import SkeletonBuilder
from video_loader import VideoLoader
from visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-based motion analysis with skeleton overlay.")
    parser.add_argument("--input", required=True, help="Path to input video (mp4, avi, mov)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target-fps", type=float, default=None, help="Normalize FPS to this value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoLoader(str(input_path), target_fps=args.target_fps)
    detector = PoseDetector()
    builder = SkeletonBuilder()
    analyzer = MotionAnalyzer()
    reports: list[MotionReport] = []

    output_video_path = output_dir / "skeleton_overlay.mp4"
    visualizer = Visualizer(output_path=output_video_path, fps=loader.meta.fps, frame_size=(loader.meta.width, loader.meta.height))

    prev_time = None
    for frame in loader.frames():
        keypoints = detector.detect(frame.rgb, loader.meta.width, loader.meta.height)
        skeleton = builder.build(keypoints)
        if prev_time is None:
            delta_t = 1.0 / loader.meta.fps
        else:
            delta_t = frame.timestamp_s - prev_time
        prev_time = frame.timestamp_s

        report = analyzer.analyze(skeleton.points_3d, delta_t)
        reports.append(report)
        visualizer.draw(frame.bgr, skeleton.points_2d, report)

    visualizer.close()
    detector.close()
    loader.release()

    writer = MathReportWriter(output_dir)
    writer.write(reports)

    logger.info("Output video saved to %s", output_video_path)
    logger.info("Math reports saved to %s", output_dir)


if __name__ == "__main__":
    main()

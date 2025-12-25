# Video Output: Skeleton-Based Motion Analysis

Production-ready Python pipeline that detects a human pose in a video, builds a 2D + pseudo-3D skeleton, analyzes motion (angles, lengths, velocity, acceleration), overlays the skeleton on the original video, and writes a detailed math report in TXT and JSON.

## Architecture (Modules)
- `video_loader.py` — OpenCV reader with FPS normalization and BGR→RGB→Grayscale conversion.
- `pose_detector.py` — MediaPipe Pose wrapper, returns keypoints with visibility scores.
- `skeleton_builder.py` — builds a connected skeleton, interpolates missing points, smooths trajectories.
- `motion_analysis.py` — computes vectors, angles, segment lengths, velocity, and acceleration.
- `visualizer.py` — draws a skeleton overlay with speed-based coloring.
- `math_report.py` — writes detailed math reports (TXT + JSON).
- `main.py` — CLI entry point orchestrating the pipeline.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py --input path/to/input.mp4 --output output_dir --target-fps 30
```

## Outputs
- `output_dir/skeleton_overlay.mp4` — video with skeleton overlay.
- `output_dir/math_report.txt` — detailed, human-readable math report.
- `output_dir/math_report.json` — structured JSON report.

## Math Report Example (excerpt)
```
Математический отчёт
=====================

Использованные формулы:
1. Длина сегмента: L = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
2. Угол между векторами: θ = arccos( (v1·v2) / (||v1|| ||v2||) )
3. Скорость: v = (p_t - p_(t-1)) / Δt
4. Ускорение: a = (v_t - v_(t-1)) / Δt

Логи вычислений по кадрам:
- Segment left_upper_arm: length = ||left_elbow - left_shoulder|| = sqrt(sum((x_i^b - x_i^a)^2))
- Angle left_elbow: arccos( (v1·v2) / (||v1||*||v2||) )
- Velocity left_elbow: v = (p_t - p_(t-1)) / Δt; Accel a = (v_t - v_(t-1)) / Δt
```

## Notes
- MediaPipe Pose provides the depth-like `z` coordinate, used as pseudo-3D.
- Smoothing uses a moving average window to reduce jitter.
- Missing points are interpolated by last valid position when visibility is low.

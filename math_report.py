from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from motion_analysis import MotionReport


class MathReportWriter:
    """Write human-readable and JSON reports for the analysis."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, reports: List[MotionReport]) -> None:
        text_path = self.output_dir / "math_report.txt"
        json_path = self.output_dir / "math_report.json"

        math_operations: List[str] = []
        for report in reports:
            math_operations.extend(report.math_operations)

        text_lines = [
            "Математический отчёт",
            "=====================",
            "",
            "Использованные формулы:",
            "1. Длина сегмента: L = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)",
            "2. Угол между векторами: θ = arccos( (v1·v2) / (||v1|| ||v2||) )",
            "3. Скорость: v = (p_t - p_(t-1)) / Δt",
            "4. Ускорение: a = (v_t - v_(t-1)) / Δt",
            "",
            "Логи вычислений по кадрам:",
            *[f"- {op}" for op in math_operations],
        ]
        text_path.write_text("\n".join(text_lines), encoding="utf-8")

        json_payload = {
            "summary": {
                "formulas": {
                    "segment_length": "L = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)",
                    "angle": "theta = arccos( (v1·v2) / (||v1|| ||v2||) )",
                    "velocity": "v = (p_t - p_(t-1)) / Δt",
                    "acceleration": "a = (v_t - v_(t-1)) / Δt",
                }
            },
            "frames": [asdict(report) for report in reports],
            "math_operations": math_operations,
        }
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

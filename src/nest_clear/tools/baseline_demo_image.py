"""合成针尖并使用生产检测与标注生成界面演示数据。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from nest_clear.baseline.session import BaselineSession, DEFAULTS


def main(root: str, state: str) -> None:
    """生成演示用两路针尖剪影图并调用生产检测和标注逻辑。"""
    images = {}
    for name in ("front", "side"):
        image = np.full((480, 640), 245, np.uint8)
        x = 326 if state == "fail" and name == "front" else 320
        cv2.line(image, (x, 120), (x, 237), 15, 5)
        images[name] = image
    obj = object.__new__(BaselineSession)
    obj.config = SimpleNamespace(
        tool_calibration=SimpleNamespace(
            detection={name: {"direction": [0, 1], "threshold": 127, "min_area": 5} for name in images}
        )
    )
    obj.events = SimpleNamespace(directory=Path(root))
    obj.cfg = DEFAULTS.copy()
    detections = obj._detect(images)
    points = {n: list(d.point_uv) for n, d in detections.items()}
    center = obj._target(images["front"])
    delta = [(points["front"][i] - center[i]) * DEFAULTS["main_um_per_px"] / 1000.0 for i in (0, 1)]
    candidate = {
        "pixels": points,
        "center": list(center),
        "delta_mm": delta,
        "distance_mm": float(np.hypot(*delta)),
        "inside": state != "fail",
        "passed": state in ("success", "confirm"),
        "p0": [125.2, 73.4, 110.0] if state in ("success", "confirm") else None,
        "click": {},
        "focus": {
            n: {
                "focused": d.focused,
                "edge_width_px": d.edge_width_px,
                "confidence": d.confidence,
                "coarse": d.coarse,
            }
            for n, d in detections.items()
        },
    }
    obj.candidate = candidate if state in ("success", "confirm", "fail") else None
    obj._annotate(images, detections)
    path = Path(root) / "baseline_demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])

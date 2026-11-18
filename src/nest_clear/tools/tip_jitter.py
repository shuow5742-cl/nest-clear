"""测量静止针尖连拍的检测抖动。"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from nest_clear.config import load_config
from nest_clear.tooltip import SilhouetteTipDetector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--view", choices=("front", "side"), default="front")
    parser.add_argument("--config", default="config/nest_clear.yaml")
    parser.add_argument("--um-per-px", type=float, default=15.2)
    args = parser.parse_args(argv)
    if not np.isfinite(args.um_per_px) or args.um_per_px <= 0:
        parser.error("--um-per-px 必须为有限正数")
    detector = SilhouetteTipDetector(**load_config(args.config, require_tools=False).tool_calibration.detection[args.view])
    points, failed = [], []
    for path in sorted(Path(args.directory).iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            failed.append(f"{path.name}: 无法读取")
            continue
        result = detector.detect(image)
        if result.point_uv is None:
            failed.append(f"{path.name}: {result.diagnostic}")
            continue
        points.append(result.point_uv)
        print(f"{path.name}: 点位={result.point_uv} focused={result.focused} edge_width={result.edge_width_px} px")
    for item in failed:
        print(f"检测失败：{item}")
    if not points:
        print("没有可统计的图像")
        return 1
    values = np.asarray(points)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    deviation = np.max(np.abs(values - mean), axis=0)
    print(f"u/v 均值={mean} px，std={std} px ({std * args.um_per_px} µm)，最大偏差={deviation} px ({deviation * args.um_per_px} µm)")
    print("抖动 std ≤ 0.3px → 0.03mm 阈值稳" if np.all(std <= .3) else "抖动 std > 0.3px → 0.03mm 阈值不稳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

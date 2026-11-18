"""用真实一代图块演示二代粗拍、精定位和对刀标注。"""
from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
from itertools import islice
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from nest_clear.config import load_config
from nest_clear.record.events import EventRecorder
from nest_clear.record.images import ImageArchive
from nest_clear.vision.fine import FineResult
from nest_clear.vision.geometry import load_calibration
from nest_clear.vision.hair_integration import measure, render, render_fine
from nest_clear.workers.vision import HairDetector

ROOT = Path(__file__).resolve().parents[3]


def make_scene(patches: Path) -> tuple[np.ndarray, list[Path]]:
    """按两代像素尺度转换原图块并拼到空白画布，不修改源文件。"""
    files = sorted(islice((p for p in patches.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")), 500))
    images = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None and min(image.shape) >= 100:
            # 暗像素较多的图块优先，用来保证演示包含真实毛发。
            images.append((float(np.percentile(image, 2)), path, image))
    images.sort(key=lambda item: item[0])
    if len(images) < 6:
        raise RuntimeError(f"{patches} 内可用图块不足 6 张")
    scene = np.full((3036, 4024), 220, np.uint8)
    used = []
    for index, (_, path, image) in enumerate(images[:16]):
        h, w = image.shape
        scale = 6.959 / 23.585
        tile = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        if tile.shape[0] > 300 or tile.shape[1] > 300:
            tile = tile[:300, :300]
        # 边缘向灰底过渡，避免拼接边框被误识别成毛。
        h, w = tile.shape
        y, x = 420 + (index // 4) * 630, 250 + (index % 4) * 930
        if y + h >= scene.shape[0] or x + w >= scene.shape[1]:
            break
        yy, xx = np.indices((h, w))
        alpha = np.minimum.reduce((xx, yy, w - 1 - xx, h - 1 - yy)).clip(0, 12) / 12.0
        scene[y:y+h, x:x+w] = np.rint(tile * alpha + 220 * (1 - alpha)).astype(np.uint8)
        used.append(path)
    return scene, used


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"无法写入 {path}")


def demo(out: Path, patches: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scene, used = make_scene(patches)
    config = load_config(ROOT / "config/nest_clear.yaml")
    calibration_path = ROOT / "config" / config.vision.calibration
    calibration = load_calibration(calibration_path)
    detector = HairDetector(config.vision.hair_detector, calibration, config.scan.z, config.vision.scale_z_bin_mm)
    candidates = detector.detect(scene)
    if len(candidates) < 6:
        raise RuntimeError(f"检测到 {len(candidates)} 个候选，无法展示六种状态；请检查图块或检测配置")
    pose = (0., 0., config.scan.z)
    observations = [measure(candidate, pose, calibration, detector.config) for candidate in candidates]
    # 取空间分散的真实检测结果，便于看清每种状态；其余候选保留算法原状态。
    chosen = [observations[0]]
    while len(chosen) < 6:
        remaining = [item for item in observations if all(item is not other for other in chosen)]
        chosen.append(max(remaining, key=lambda item: min(np.linalg.norm(np.asarray(item.candidate.pick_uv) - np.asarray(other.candidate.pick_uv)) for other in chosen)))
    statuses = ("eligible", "rejected", "failed", "fine_failed", "attempted", "other_view")
    for index, (item, status) in enumerate(zip(chosen, statuses), 1):
        item.status = status
        item.reason = "small" if status == "rejected" else None
        item.target_id = f"0001-1-{index:03d}"
    wall = datetime.now().astimezone()
    capture_id = f"{wall:%Y%m%d_%H%M%S}#000001"
    info = {"capture_id": capture_id, "wall": wall.isoformat(timespec="seconds"),
            "tray": "2026-01-01_0001", "view": 1, "shot": "c01", "task_id": 17,
            "eligible": sum(item.status == "eligible" for item in observations),
            "failed": sum(item.status == "failed" for item in observations),
            "um_per_px": round(detector.config.um_per_px, 3)}
    save_image(out / "coarse.jpg", render(scene, observations, targets=tuple(chosen), info=info))
    selected = chosen[0]
    dx = round(calibration.K[0, 2] - selected.candidate.pick_uv[0])
    dy = round(calibration.K[1, 2] - selected.candidate.pick_uv[1])
    fine_scene = cv2.warpAffine(scene, np.float32([[1, 0, dx], [0, 1, dy]]),
                                (scene.shape[1], scene.shape[0]), borderValue=220)
    original = selected.candidate
    shifted = replace(original, pick_uv=(original.pick_uv[0] + dx, original.pick_uv[1] + dy),
                      identity_uv=(original.identity_uv[0] + dx, original.identity_uv[1] + dy),
                      polygon=tuple((x + dx, y + dy) for x, y in original.polygon),
                      bbox=(original.bbox[0] + dx, original.bbox[1] + dy,
                            original.bbox[2] + dx, original.bbox[3] + dy))
    fine_selected = measure(shifted, pose, calibration, detector.config)
    fine_selected.target_id = selected.target_id
    fine_selected.status = "eligible"
    fine = FineResult("ok", candidate=shifted, pick_uv=shifted.pick_uv,
                      target_xyz=fine_selected.target.xyz, identity=fine_selected.identity)
    fine_info = {"attempt_id": f"{selected.target_id}#a1", "task_id": 18,
                 "tool1": tuple(round(v, 3) for v in fine_selected.target.xyz),
                 "tool2": tuple(round(v + .1, 3) for v in fine_selected.target.xyz)}
    coarse_uv = (shifted.pick_uv[0] + 35, shifted.pick_uv[1] + 25)
    save_image(out / "fine.jpg", render_fine(fine_scene, [fine_selected], fine, pose, calibration,
                                               config.vision.fine_box_half_mm, info=fine_info, coarse_uv=coarse_uv))
    # 走生产归档的对刀 overlay 绘制路径，生成两视图质量信息。
    tip = np.full((960, 1280), 220, np.uint8)
    cv2.line(tip, (580, 150), (640, 470), 20, 12)
    cv2.circle(tip, (640, 470), 6, 20, -1)
    with tempfile.TemporaryDirectory(prefix="nest-clear-demo-") as temporary:
        events = EventRecorder(temporary, run=True)
        events.begin_request(19, 110)
        archive = ImageArchive(events, True, "jpg", 95, 30)
        sequence, cid = events.next_capture()
        raw = archive.capture_raw(tip, sequence=sequence, capture_id=cid, kind="calib_front", tray="0001", view=0, shot="i04", task_id=19)
        overlay = archive.calibration(tip, 1, 19, 4, "front", (640, 470), (660, 480), raw,
                                      focus={"front": {"edge_width_px": 2.8, "focused": True},
                                             "side": {"edge_width_px": 3.1, "focused": True}})
        archive.close()
        if not overlay:
            raise RuntimeError("对刀标注图归档失败")
        shutil.copyfile(Path(temporary) / overlay, out / "calib.jpg")
    print(f"读取 {len(used)} 张图块，检测到 {len(candidates)} 个候选；输出目录：{out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成真实图块的标注演示")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--patches", type=Path, default=Path.home() / "pluck-ds/patches")
    args = parser.parse_args()
    demo(args.out, args.patches)


if __name__ == "__main__":
    main()

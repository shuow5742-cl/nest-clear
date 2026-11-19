"""将毛候选投影到工作平面、应用物理门并绘制操作画面。"""
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .detector import ModuleTarget, PixelTarget
from .geometry import Calibration, pixel_to_module
from .hair import HairCandidate, HairDetectConfig


@dataclass
class Observation:
    """关联一个候选的扎点、身份、物理尺寸和当前帧状态。"""

    target: ModuleTarget  # 扎点像素及其模组 XYZ，供工具补偿后下发
    identity: tuple[float, float, float]  # bbox 中心投影后的模组 XYZ，供跨帧匹配
    candidate: HairCandidate  # 原始轮廓、暗度和像素尺寸
    length: float  # 局部尺度近似的长度，mm
    area: float  # 投影轮廓的面积，mm²；stub 用分数作为排序键
    reason: str | None  # 物理门拒绝原因；通过时为 None
    status: str = "eligible"  # eligible、rejected、failed 或 attempted


def measure(c: HairCandidate, pose: tuple[float, float, float], calibration: Calibration,
            cfg: HairDetectConfig) -> Observation:
    """投影轮廓求面积；长度用中心水平、垂直局部尺度均值近似。"""
    def project(point: tuple[float, float]) -> tuple[float, float, float]:
        """用拍照时实时模组坐标把一个像素投影到工作平面。"""
        return pixel_to_module(*point, pose, calibration)

    center = project(c.identity_uv)
    x = project((c.identity_uv[0] + 1, c.identity_uv[1]))
    y = project((c.identity_uv[0], c.identity_uv[1] + 1))
    scale = (math.dist(center[:2], x[:2]) + math.dist(center[:2], y[:2])) / 2
    points = [project(p) for p in c.polygon]
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2
    length = c.length_px * scale
    width = area / length if length > 0 else 0
    reason = ("short" if length < cfg.min_length_mm else
              "small" if area < cfg.min_area_mm2 else
              "thin" if width < cfg.min_width_mm else
              "pale" if (c.gray_ratio is not None and c.dark_frac is not None and
                         (c.gray_ratio > cfg.max_gray_ratio or c.dark_frac < cfg.min_dark_frac)) else None)
    pixel = PixelTarget(*c.pick_uv, 1.0, c)
    return Observation(ModuleTarget(pixel, project(c.pick_uv)), center, c, length, area, reason,
                       "rejected" if reason else "eligible")


# OpenCV BGR：绿=要挑，黄=物理门拒绝，红=挑取失败；青=曾判成功但又出现（仍跳过并告警）。
COLORS = {"eligible": (0, 255, 0), "rejected": (0, 255, 255),
          "failed": (0, 0, 255), "fine_failed": (0, 165, 255), "attempted": (255, 255, 0), "other_view": (128, 128, 128), "out_of_range": (255, 0, 0)}
LABELS = {"eligible": "PICK", "rejected": "skip", "failed": "FAILED", "fine_failed": "FINE FAIL", "attempted": "PENDING", "other_view": "OTHER", "out_of_range": "OUT"}


def _chinese(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], size: int) -> None:
    """在小画布绘制中文，避免转换整张高分辨率图片。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = next(path for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
                            "C:/Windows/Fonts/msyh.ttc") if Path(path).exists())
        font = ImageFont.truetype(font_path, size)
        x, y = origin
        x, y = max(0, x), max(0, y)
        width, height = min(image.shape[1]-x, size * (len(text)+1)), min(image.shape[0]-y, size*2)
        if width <= 0 or height <= 0:
            return
        canvas = Image.fromarray(cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(canvas).text((0, 0), text, font=font, fill=tuple(reversed(color)))
        image[y:y+height, x:x+width] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError, StopIteration):
        pass


def _legend(image: np.ndarray, entries: tuple[tuple[str, tuple[int, int, int]], ...], factor: float) -> None:
    """在右下角放置带色块的图例，浅色和深色画面均可阅读。"""
    if image.shape[1] < 200 or image.shape[0] < 150:
        return
    font = max(.35, .43 * factor)
    thickness = max(1, round(factor))
    step = max(16, round(21 * factor))
    swatch = max(8, round(12 * factor))
    text_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0] for label, _ in entries)
    width = min(image.shape[1], text_width + swatch + round(25 * factor))
    height = min(image.shape[0], step * len(entries) + round(12 * factor))
    left, top = image.shape[1] - width, image.shape[0] - height
    cv2.rectangle(image, (left, top), (image.shape[1] - 1, image.shape[0] - 1), (32, 32, 32), -1)
    for index, (label, color) in enumerate(entries):
        y = top + round(8 * factor) + step * index
        x = left + round(6 * factor)
        cv2.rectangle(image, (x, y), (x + swatch, y + swatch), color, -1)
        cv2.putText(image, label, (x + swatch + round(6 * factor), y + swatch),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thickness, cv2.LINE_AA)


COARSE_LEGEND = (("PICK", COLORS["eligible"]), ("GATE", COLORS["rejected"]),
                 ("FAILED", COLORS["failed"]), ("FINE FAIL", COLORS["fine_failed"]),
                 ("PENDING", COLORS["attempted"]), ("OTHER", COLORS["other_view"]), ("OUT", COLORS["out_of_range"]))


def render(image: np.ndarray, observations: list[Observation], selected: Observation | None = None, ignore_regions: tuple[tuple[float, ...], ...] = (), targets: tuple[Observation, ...] = (), *, info: dict | None = None, legend: bool = True, header: bool = True, pick_region=None, pose=None, calibration=None) -> np.ndarray:
    """按画面尺寸缩放标注，显示每个候选和本次选中目标的扎点。"""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    factor = max(0.5, image.shape[1] / 1280.0)
    font_scale = 0.38 * factor
    thickness = max(1, round(factor))
    line_width = max(1, round(2 * factor))
    for x1, y1, x2, y2 in ignore_regions:
        x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))
        dash = max(4, round(10 * factor))
        for x in range(x1, x2, 2 * dash):
            cv2.line(image, (x, y1), (min(x + dash, x2), y1), (128, 128, 128), 1)
            cv2.line(image, (x, y2), (min(x + dash, x2), y2), (128, 128, 128), 1)
        for y in range(y1, y2, 2 * dash):
            cv2.line(image, (x1, y), (x1, min(y + dash, y2)), (128, 128, 128), 1)
            cv2.line(image, (x2, y), (x2, min(y + dash, y2)), (128, 128, 128), 1)
    if pick_region is not None and pose is not None and calibration is not None:
        (xmin, xmax), (ymin, ymax) = pick_region
        normal, plane_point = calibration.plane_normal, calibration.plane_point
        if abs(normal[2]) < 1e-12:
            raise ValueError("工作平面无法按 XY 投影可挑范围")
        corners = np.array([(x, y, plane_point[2] - (normal[0] * (x - plane_point[0]) + normal[1] * (y - plane_point[1])) / normal[2])
                            for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))])
        camera = (corners - (np.asarray(pose) + calibration.translation)) @ calibration.rotation
        if np.all(camera[:, 2] > 1e-12):
            uv, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), calibration.K, calibration.distortion)
            points = uv.reshape(-1, 2)
            if np.isfinite(points).all() and np.max(np.abs(points)) < 1e7:
                points = np.rint(points).astype(int)
                dash = max(4, round(10 * factor))
                for a, b in zip(points, np.roll(points, -1, axis=0)):
                    visible, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), tuple(a), tuple(b))
                    if visible:
                        start, end = np.array(start), np.array(end)
                        length = float(np.linalg.norm(end - start))
                        if length:
                            for d in np.arange(0, length, 2 * dash):
                                p1 = start + (end - start) * (d / length)
                                p2 = start + (end - start) * (min(d + dash, length) / length)
                                cv2.line(image, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), (255, 0, 0), line_width)
    for observation in observations:
        candidate = getattr(observation, "candidate", None)
        if candidate is None:
            continue
        status = getattr(observation, "status", "other_view")
        color = COLORS.get(status, COLORS["other_view"])
        poly = np.asarray(candidate.polygon, dtype=np.int32)
        if len(poly) >= 3 and observation.status != "other_view":
            layer = image.copy()
            cv2.fillPoly(layer, [poly], color)
            image = cv2.addWeighted(layer, 0.2, image, 0.8, 0)
        x1, y1, x2, y2 = map(round, candidate.bbox)
        if observation.status == "fine_failed":
            dash = max(4, round(10 * factor))
            for x in range(x1, x2, 2 * dash):
                cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, line_width)
                cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, line_width)
            for y in range(y1, y2, 2 * dash):
                cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, line_width)
                cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, line_width)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if observation.status == "other_view" else line_width)
        label = LABELS.get(status, "OTHER")
        if observation.reason and observation.status == "rejected":
            label += f" {observation.reason}"
        target_id = getattr(observation, "target_id", None)
        target_tail = str(target_id).split("-")[-1] if target_id else "--"
        label += f" #{target_tail} A={observation.area:.2f}mm2"
        baseline = max(round(28 * factor), y1 - round(6 * factor))
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        left = max(0, min(x1, image.shape[1] - text_width - 1))
        cv2.putText(image, label, (left, baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + max(2, round(2*factor)), cv2.LINE_AA)
        cv2.putText(image, label, (left, baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        for index, target in enumerate(targets, 1):
            if index <= 10 and observation is target:
                cv2.putText(image, str(index), (x1, min(image.shape[0]-1, y1 + round(20*factor))), cv2.FONT_HERSHEY_SIMPLEX, 0.6*factor, (255, 255, 255), line_width)
        point = tuple(map(round, candidate.pick_uv))
        cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, max(7, round(18 * factor)), line_width)
        if observation.status == "out_of_range":
            _chinese(image, "范围外", (x1, y2 + 2), color, max(12, round(20*factor)))
        if observation.status == "fine_failed":
            _chinese(image, "精定位失败", (x1, y2 + 2), color, max(12, round(20*factor)))
        if observation is selected:
            cv2.circle(image, point, max(3, round(6 * factor)), (0, 0, 255), line_width)
    if header and (info is not None or (image.shape[1] >= 200 and image.shape[0] >= 150)):
        info = info or {}
        font = max(.45, .55 * factor)
        def display(key: str) -> str:
            value = info.get(key)
            return "--" if value is None or value == "" else str(value)
        lines = [f"CAPTURE {display('capture_id')}", f"TIME {display('wall')}",
                 f"TRAY {display('tray')}  VIEW {display('view')}  SHOT {display('shot')}  TASK {display('task_id')}",
                 f"ELIGIBLE {display('eligible')}  RED {display('failed')}  SCALE {display('um_per_px')} um/px"]
        width = min(image.shape[1], max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0] for line in lines) + round(20*factor))
        cv2.rectangle(image, (0, 0), (width, min(image.shape[0]-1, round(95*factor))), (32, 32, 32), -1)
        for i, text in enumerate(lines, 1):
            cv2.putText(image, text, (round(10*factor), round((22*i)*factor)), cv2.FONT_HERSHEY_SIMPLEX, font, (255,255,255), thickness, cv2.LINE_AA)
    if legend:
        _legend(image, COARSE_LEGEND, factor)
    return image


def render_fine(image: np.ndarray, observations: list[Observation], result: object,
                pose: tuple[float, float, float], calibration: Calibration, half_mm: float,
                *, info: dict | None = None, coarse_uv: tuple[float, float] | None = None) -> np.ndarray:
    """投影工作面精定位框，标出选中轮廓与裁剪后的新扎点。"""
    selected = next ( (o for o in observations if o.candidate is result.candidate), None)
    output = render(image, observations, selected, legend=False, header=False)
    center = pixel_to_module(calibration.K[0, 2], calibration.K[1, 2], pose, calibration)
    origin = np.asarray(pose) + calibration.translation
    corners = np.array([[center[0] + x, center[1] + y, calibration.plane_z]
                        for x, y in ((-half_mm, -half_mm), (half_mm, -half_mm),
                                     (half_mm, half_mm), (-half_mm, half_mm))])
    camera = (corners - origin) @ calibration.rotation
    uv, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), calibration.K, calibration.distortion)
    factor = max(.5, image.shape[1] / 1280)
    line_width = max(1, round(2*factor))
    cv2.polylines(output, [np.rint(uv.reshape(-1, 2)).astype(np.int32)], True, (255, 0, 255), line_width)
    cv2.drawMarker(output, (round(calibration.K[0, 2]), round(calibration.K[1, 2])), (255,255,255), cv2.MARKER_CROSS, max(7, round(18*factor)), max(1, round(2*factor)))
    if coarse_uv is not None:
        cv2.circle(output, tuple(map(round, coarse_uv)), max(4, round(10*factor)), (0, 255, 255), max(1, round(2*factor)))
    if result.pick_uv is not None:
        cv2.drawMarker(output, tuple(map(round, result.pick_uv)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, max(7, round(18*factor)), line_width)
        if result.target_xyz is not None:
            cv2.putText(output, str(tuple(round(v, 3) for v in result.target_xyz)), tuple(map(round, result.pick_uv)), cv2.FONT_HERSHEY_SIMPLEX, .6*factor, (255,255,0), max(1,round(2*factor)), cv2.LINE_AA)
    if info:
        tool1, tool2 = info.get("tool1"), info.get("tool2")
        lines = [str(info.get("attempt_id") or "--"), f"Task_ID={info.get('task_id')}"]
        if tool1 is not None and tool2 is not None:
            lines.append(f"T1={tool1} T2={tool2}")
        else:
            reason = info.get("not_sent_reason") or getattr(result, "status", "unknown")
            lines.append(f"NOT SENT: {reason}")
        panel_width = min(output.shape[1] - 1, max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .55*factor, max(1, round(factor)))[0][0] for text in lines) + round(20*factor))
        panel_height = min(output.shape[0] - 1, round((110 if tool1 is None or tool2 is None else 85)*factor))
        cv2.rectangle(output, (0, 0), (panel_width, panel_height), (32, 32, 32), -1)
        for i, text in enumerate(lines, 1):
            cv2.putText(output, text, (round(10*factor), round(25*i*factor)), cv2.FONT_HERSHEY_SIMPLEX, .55*factor, (255,255,255), max(1,round(factor)), cv2.LINE_AA)
        if tool1 is None or tool2 is None:
            reason = info.get("not_sent_reason") or getattr(result, "status", "unknown")
            reasons = {"fine_empty": "空框", "fine_rejected": "门拒", "fine_point_outside": "点在框外", "fine_out_of_range": "超出可挑范围",
                       "ok": "等待下发", "reply_failed": "应答未提交"}
            _chinese(output, f"未下发：{reasons.get(reason, reason)}", (round(10*factor), round(80*factor)), (255,255,255), max(12,round(18*factor)))
    _legend(output, (("FINE BOX", (255,0,255)), ("PRINCIPAL", (255,255,255)),
                     ("COARSE PICK", (0,255,255)), ("SENT PICK", (255,255,0))), factor)
    return output

"""在后台执行检测、平面投影和跨帧身份判定，向行为树返回本帧结果。"""
from collections import Counter
import logging
import math
import time
from typing import Any

import cv2
import numpy as np
from autoweaver.worker import AsyncRequestWorker

from nest_clear.config import VisionConfig
from nest_clear.vision.detector import Detector, ModuleTarget, PixelTarget, postprocess
from nest_clear.vision.geometry import Calibration, pixel_to_module
from nest_clear.vision.hair import HairCandidate, HairDetectConfig, detect_candidates, effective_pixel_values, scaled
from nest_clear.vision.hair_integration import Observation, measure

logger = logging.getLogger(__name__)


class HairDetector:
    """缓存一次缩放后的毛检测参数，为视觉 Worker 提供全部候选。"""

    def __init__(self, config: HairDetectConfig) -> None:
        """按实际像素尺度换算一次并记录生效参数。"""
        self.config = scaled(config)
        logger.info("毛检测 effective_pixel_values：%s", effective_pixel_values(self.config, already_scaled=True))

    def detect(self, image: np.ndarray) -> list[HairCandidate]:
        """将单通道图转为 BGR 后执行完整检测流水线。"""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_candidates(image, self.config)


def stub_observation(pixel: PixelTarget, pose: tuple[float, float, float], calibration: Calibration) -> Observation:
    """将测试用像素目标适配为观察结果；分数作为排序键，不应用物理门。"""
    xyz = pixel_to_module(pixel.u, pixel.v, pose, calibration)
    candidate = HairCandidate(
        (pixel.u, pixel.v), None, (), (pixel.u - 2, pixel.v - 2, pixel.u + 2, pixel.v + 2),
        (pixel.u, pixel.v), 0, 0, None, None, "stub", "stub",
    )
    return Observation(ModuleTarget(pixel, xyz), xyz, candidate, 0, pixel.score, None)


class VisionWorker(AsyncRequestWorker):
    """为每帧计算物理门、挑后结果和已尝试过滤，不阻塞行为树 tick。"""

    name = "vision"

    def __init__(self, detector: Detector | HairDetector, calibration: Calibration, radius: float,
                 maximum: int, config: VisionConfig | None = None) -> None:
        """保存检测器及投影、筛选参数，兼容旧版目标后处理调用。"""
        super().__init__()
        self.detector = detector
        self.calibration = calibration
        self.radius = radius
        self.maximum = maximum
        self.config = config

    def execute(self, payload: dict[str, Any]) -> Any:
        """检测当前帧并以模组坐标复核待挑目标、过滤本盘已尝试目标。"""
        start = time.perf_counter()
        pixels = self.detector.detect(payload["image"])
        pose = payload["pose"]
        if self.config is None:
            targets = [ModuleTarget(p, pixel_to_module(p.u, p.v, pose, self.calibration)) for p in pixels]
            return postprocess(targets, self.radius, self.maximum)
        def ignored(p: Any) -> bool:
            """按候选包围盒中心屏蔽现场针区。"""
            if isinstance(p, HairCandidate):
                x1, y1, x2, y2 = p.bbox
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
            else:
                x, y = p.u, p.v
            return any(a <= x <= c and b <= y <= d for a, b, c, d in self.config.ignore_regions_px)

        ignored_count = sum(ignored(p) for p in pixels)
        pixels = [p for p in pixels if not ignored(p)]
        observations = [
            measure(p, pose, self.calibration, self.config.hair_detector)
            if isinstance(p, HairCandidate) else stub_observation(p, pose, self.calibration)
            for p in pixels
        ]
        pending = payload.get("pending")
        result = None
        if pending:
            nearest = min(observations, key=lambda o: math.dist(o.identity[:2], pending["identity"][:2]), default=None)
            distance = math.dist(nearest.identity[:2], pending["identity"][:2]) if nearest else None
            failed = distance is not None and distance <= self.config.review_radius_mm
            result = {**pending, "success": not failed, "current_identity": nearest.identity if nearest else None,
                      "distance_mm": distance}
        attempted = list(payload.get("attempted", []))
        if result:
            attempted.append({"identity": pending["identity"], "success": result["success"]})
        warnings = []
        for observation in observations:
            match = next((a for a in attempted if math.dist(observation.identity[:2], a["identity"][:2])
                          <= self.config.skip_radius_mm), None)
            if match:
                observation.status = "attempted" if match["success"] else "failed"
                if match["success"]:
                    warnings.append("成功目标再次出现")
        path = payload.get("scan_path")
        scan = payload.get("scan")
        if self.config.responsibility_zone and path and scan is not None:
            center = pixel_to_module(self.calibration.K[0, 2], self.calibration.K[1, 2], pose, self.calibration)
            centers = [(center[0] + p[0] - scan[0], center[1] + p[1] - scan[1]) for p in path]
            for observation in observations:
                if observation.status == "eligible" and not owns_target(observation.target.xyz[:2], center[:2], centers):
                    observation.status = "other_view"
        eligible = sorted((o for o in observations if o.status == "eligible"), key=lambda o: o.area, reverse=True)
        return {
            "observations": observations, "eligible": eligible, "result": result, "warnings": warnings,
            "count": len(observations), "eligible_count": len(eligible),
            "rejections": dict(Counter(o.reason for o in observations if o.reason)),
            "ignored_count": ignored_count, "other_view_count": sum(o.status == "other_view" for o in observations),
            "failed_count": sum(o.status == "failed" for o in observations),
            "elapsed_ms": (time.perf_counter() - start) * 1000,
        }


def owns_target(target: tuple[float, ...], center: tuple[float, ...], centers: list[tuple[float, float]]) -> bool:
    """相等时本视野也负责；距离比较使用扎点的模组坐标。"""
    distance = math.dist(target, center)
    return all(distance <= math.dist(target, other) + 1e-9 for other in centers)

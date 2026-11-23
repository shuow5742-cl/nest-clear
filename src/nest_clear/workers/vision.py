"""在后台执行检测、平面投影和跨帧身份判定，向行为树返回本帧结果。"""
from collections import Counter
import logging
import math
import time
import threading
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from autoweaver.worker import AsyncRequestWorker

from nest_clear.config import VisionConfig
from nest_clear.vision.detector import Detector, ModuleTarget, PixelTarget, postprocess
from nest_clear.vision.geometry import Calibration, pixel_to_module, scale_um_per_px
from nest_clear.guard import GuardError
from nest_clear.vision.hair.scaling import INVENTORY
from nest_clear.vision.fine import FineResult, select_fine_target
from nest_clear.vision.hair import HairCandidate, HairDetectConfig, detect_candidates, effective_pixel_values, scaled
from nest_clear.vision.hair_integration import Observation, measure

logger = logging.getLogger(__name__)


class HairDetector:
    """缓存一次缩放后的毛检测参数，为视觉 Worker 提供全部候选。"""

    def __init__(self, config: HairDetectConfig, calibration: Calibration | None = None, camera_z: float = 0., scale_z_bin_mm: float = 0.1) -> None:
        self.reference = config
        self.calibration = calibration
        if not math.isfinite(scale_z_bin_mm) or scale_z_bin_mm <= 0:
            raise ValueError("scale_z_bin_mm 必须为正有限数")
        self.scale_z_bin_mm = scale_z_bin_mm
        self.cache = {}
        self._logged_keys = set()
        self.config = self.for_z(camera_z) if calibration is not None else scaled(config)
        for name, (_, category, _) in INVENTORY.items():
            before, after = getattr(config, name), getattr(self.config, name)
            kind = "面积" if category == "area" else "长度" if category in ("length", "kernel", "fiber_kernel", "radius_optional") else "无关"
            logger.info("参数 %s：%s → %s（类型：%s）", name, before, after, kind)
            k = config.reference_um_per_px / self.config.um_per_px
            floor_hit = (category == "kernel" and before * k < 3) or (category == "fiber_kernel" and before * config.fiber_ds * k / self.config.fiber_ds < 3)
            if category in ("length", "area", "kernel", "fiber_kernel") and isinstance(after, (int, float)) and (floor_hit or after <= (4 if category == "area" else 2)):
                logger.warning("参数 %s 换算后仅 %s %s，低分辨率下可能失效，需用真图复核", name, after, "px²" if category == "area" else "px")

    def for_z(self, z: float) -> HairDetectConfig:
        if not math.isfinite(z):
            raise GuardError("VISION-SCALE", f"拍照 Z 非有限值：{z}", "视觉尺度", "检查 PLC 实时坐标。")
        key = math.floor(z / self.scale_z_bin_mm + 0.5)
        bin_z = key * self.scale_z_bin_mm
        if key not in self.cache:
            scale = scale_um_per_px(self.calibration, self.calibration.plane_z, bin_z)
            expected = self.reference.expected_um_per_px
            if abs(scale / expected - 1) > self.reference.scale_tolerance:
                raise GuardError("VISION-SCALE", f"标定推出的尺度 {scale:.3f} µm/px 与配置期望 {expected:.3f} 不符", f"拍照 Z={z}", "检查标定文件/工作距离/配置。")
            if key not in self._logged_keys and len(self._logged_keys) < 16:
                logger.info("拍照 Z 分档=%s mm，标定尺度 %.5f µm/px", bin_z, scale)
                self._logged_keys.add(key)
            if len(self.cache) >= 16:
                del self.cache[next(iter(self.cache))]
            self.cache[key] = scaled(replace(self.reference, um_per_px=scale))
        return self.cache[key]

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
        self._calibration_lock = threading.RLock()
        self.detector = detector
        self.calibration = calibration
        self.radius = radius
        self.maximum = maximum
        self.config = config

    def update_calibration(self, calibration: Calibration) -> None:
        with self._calibration_lock:
            self.calibration = calibration
            if isinstance(self.detector, HairDetector):
                self.detector.calibration = calibration
                self.detector.cache.clear()
                self.detector._logged_keys.clear()

    def execute(self, payload: dict[str, Any]) -> Any:
        with self._calibration_lock:
            return self._execute(payload)

    def _execute(self, payload: dict[str, Any]) -> Any:
        """检测当前帧并以模组坐标复核待挑目标、过滤本盘已尝试目标。"""
        start = time.perf_counter()
        pose = payload["pose"]
        cycle = payload.get("cycle")
        if getattr(cycle, "film_mode", False) and payload.get("work_action") in ("coarse", "fine"):
            return self._film_detect(payload, start)
        if isinstance(self.detector, HairDetector) and self.detector.calibration is not None:
            self.detector.config = self.detector.for_z(pose[2])
        pixels = self.detector.detect(payload["image"])
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
        if payload.get("work_action") in ("coarse", "fine"):
            cycle = payload["cycle"]
            path = payload.get("scan_path")
            scan = payload.get("scan")
            if payload["work_action"] == "coarse" and self.config.responsibility_zone and path and scan is not None:
                center = pixel_to_module(self.calibration.K[0, 2], self.calibration.K[1, 2], pose, self.calibration)
                centers = [(center[0] + p[0] - scan[0], center[1] + p[1] - scan[1]) for p in path]
                for observation in observations:
                    if observation.status == "eligible" and not owns_target(observation.target.xyz[:2], center[:2], centers):
                        observation.status = "other_view"
            result = {"observations": observations}
            if payload["work_action"] == "fine":
                by_candidate = {id(o.candidate): o for o in observations}
                result["fine_result"] = select_fine_target(
                    payload["image"], [o.candidate for o in observations], pose, self.calibration,
                    box_half_mm=self.config.fine_box_half_mm,
                    is_attempted=lambda c: cycle.is_attempted(by_candidate[id(c)].identity),
                    passes_gate=lambda c: by_candidate[id(c)].reason is None,
                    cfg=self.detector.config if isinstance(self.detector, HairDetector) else self.config.hair_detector,
                )
            result["elapsed_ms"] = (time.perf_counter() - start) * 1000
            result["um_per_px"] = (self.detector.config.um_per_px if isinstance(self.detector, HairDetector)
                                  else scale_um_per_px(self.calibration, self.calibration.plane_z, pose[2]))
            return result
        result = None
        warnings = []
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

    def _film_detect(self, payload, start):
        from nest_clear.tooloffset.detection import detect
        cycle, pose = payload['cycle'], payload['pose']
        calibration = getattr(cycle, 'film_calibration', None)
        non_pickable_labels = getattr(cycle, 'film_non_pickable', None)
        if calibration is None or non_pickable_labels is None:
            raise GuardError('FILM-CONFIG', '菲林模式缺少平面标定或不可去目标配置', '菲林目标识别', '检查本盘测试模式配置。')
        observations = []
        for film in detect(payload['image']):
            u, v = film.ring_uv
            xyz = pixel_to_module(u, v, pose, calibration)
            marker = np.asarray(film.marker)
            size = max(8., float(np.linalg.norm(marker[1] - marker[0])) * 2)
            candidate = HairCandidate((u, v), None, (), (u-size, v-size, u+size, v+size),
                                      (u, v), 0, 0, None, None, 'film', 'film')
            non_pickable = film.label in non_pickable_labels
            obs = Observation(ModuleTarget(PixelTarget(u, v, 1., film), xyz), xyz,
                              candidate, 0., 1., '不可去' if non_pickable else None,
                              'rejected' if non_pickable else 'eligible')
            obs.film_label = film.label
            observations.append(obs)
        result = {'observations': observations, 'elapsed_ms': (time.perf_counter()-start)*1000}
        if payload['work_action'] == 'fine':
            coarse = cycle.current_target()
            label = getattr(coarse, 'film_label', None)
            choices = [o for o in observations if label is not None and getattr(o, 'film_label', None) == label and o.status == 'eligible']
            selected = min(choices, key=lambda o: math.dist(o.identity[:2], coarse.identity[:2]), default=None)
            result['fine_result'] = (FineResult('ok', selected.candidate, selected.candidate.pick_uv,
                                                 selected.target.xyz, identity=selected.identity,
                                                 box_candidate_count=len(choices)) if selected else FineResult('fine_empty'))
        return result


def owns_target(target: tuple[float, ...], center: tuple[float, ...], centers: list[tuple[float, float]]) -> bool:
    """相等时本视野也负责；距离比较使用扎点的模组坐标。"""
    distance = math.dist(target, center)
    return all(distance <= math.dist(target, other) + 1e-9 for other in centers)

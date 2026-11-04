"""将检测与几何计算封装为异步视觉 Worker。"""
from typing import Any

from nest_clear.vision.detector import Detector, ModuleTarget, postprocess
from nest_clear.vision.geometry import Calibration, pixel_to_module
from autoweaver.worker import AsyncRequestWorker


class VisionWorker(AsyncRequestWorker):
    """检测和换算可能耗时，在后台执行并显式发布完成状态。"""
    name = "vision"

    def __init__(self, detector: Detector, calibration: Calibration, radius: float, maximum: int) -> None:
        """保存检测器、标定和每视野筛选参数。"""
        super().__init__()
        self.detector = detector
        self.calibration = calibration
        self.radius = radius
        self.maximum = maximum

    def execute(self, payload: dict[str, Any]) -> Any:
        """检测像素并结合实时模组位置换算、排序和去重。"""
        pixels = self.detector.detect(payload["image"])
        targets = [ModuleTarget(p, pixel_to_module(p.u, p.v, payload["pose"], self.calibration)) for p in pixels]
        return postprocess(targets, self.radius, self.maximum)

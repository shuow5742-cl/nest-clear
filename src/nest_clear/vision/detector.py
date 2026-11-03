"""定义检测器接口与按实际毫米距离筛选的目标后处理。"""
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from nest_clear.guard import GuardError


@dataclass(frozen=True)
class PixelTarget:
    """保存检测像素、置信度及可选算法信息。"""
    u: float
    v: float
    score: float
    info: Any = None


@dataclass(frozen=True)
class ModuleTarget:
    """将像素检测与换算后的模组坐标绑定，供筛选和挑取使用。"""
    pixel: PixelTarget
    xyz: tuple[float, float, float]


class Detector(Protocol):
    """描述图像到像素目标列表的算法接口。"""
    def detect(self, image: np.ndarray) -> list[PixelTarget]:
        """检测单张图像并返回像素目标。"""
        ...


class StubDetector:
    """返回预设目标，空列表表示当前画面没有目标。"""
    def __init__(self, targets: list[PixelTarget] | None = None) -> None:
        """复制配置的目标，避免外部修改影响检测。"""
        self.targets = list(targets or [])

    def detect(self, image: np.ndarray) -> list[PixelTarget]:
        """返回预设像素目标，供无硬件集成测试使用。"""
        return list(self.targets)


def postprocess(targets: list[ModuleTarget], radius_mm: float, maximum: int) -> list[ModuleTarget]:
    """按分数降序保留目标，并按模组 XY 毫米距离去重。"""
    if not math.isfinite(radius_mm) or radius_mm < 0 or type(maximum) is not int or maximum < 1:
        raise GuardError("DETECTOR-CONFIG", f"去重半径或数量无效：{radius_mm}, {maximum}", "目标后处理", "检查 detector 配置。")
    for target in targets:
        try:
            valid = len(target.xyz) == 3 and all(math.isfinite(x) for x in (*target.xyz, target.pixel.u, target.pixel.v, target.pixel.score)) and 0 <= target.pixel.score <= 1
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise GuardError("DETECTOR-OUTPUT", f"检测目标无效：{target!r}", "目标后处理", "检查检测器的像素、分数及几何结果。")
    selected = []
    for target in sorted(targets, key=lambda item: item.pixel.score, reverse=True):
        if all(math.hypot(target.xyz[0] - other.xyz[0], target.xyz[1] - other.xyz[1]) > radius_mm for other in selected):
            selected.append(target)
            if len(selected) == maximum:
                break
    return selected

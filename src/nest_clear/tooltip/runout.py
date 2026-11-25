"""针尖静态多次停位的二维跳动量计算；不依赖界面或设备。"""

import copy
import json

import numpy as np


def _json_value(value):
    """将 numpy 值转换为普通 JSON 值，并验证调用方的元数据。"""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"无法 JSON 序列化: {type(item).__name__}")

    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _circle_diameter(points):
    if len(points) < 3:
        return None
    xy = np.asarray(points, dtype=float)
    # 中心化后求解，减少坐标原点偏大造成的数值误差。
    origin = xy.mean(axis=0)
    p = xy - origin
    a = np.column_stack((2 * p, np.ones(len(p))))
    if np.linalg.matrix_rank(a) < 3:
        return None
    solution, _, _, singular = np.linalg.lstsq(a, (p * p).sum(axis=1), rcond=None)
    if singular[-1] < singular[0] * 1e-8:
        return None
    center = solution[:2]
    radius_sq = solution[2] + center @ center
    if radius_sq <= 0:
        return None
    angles = np.sort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    # 没有跨过半圆的采样无法可靠地约束完整圆。
    if np.max(gaps) >= np.pi:
        return None
    return float(2 * np.sqrt(radius_sq))


class RunoutSession:
    def __init__(self, max_mm=0.05, stable_px=1.0, stable_presses=3, stable_delta_um=2.0):
        if max_mm < 0 or stable_px < 0 or stable_presses < 1 or stable_delta_um < 0:
            raise ValueError("会话阈值无效")
        self.max_mm = float(max_mm)
        self.stable_px = float(stable_px)
        self.stable_presses = int(stable_presses)
        self.stable_delta_um = float(stable_delta_um)
        self._presses = []

    def add_press(self, front_uv, side_uv, front_map, side_map, meta=None):
        front, side = list(front_uv), list(side_uv)
        if not front or len(front) != len(side):
            raise ValueError("两相机必须提供数量相同且非空的帧")
        for i, (f, s) in enumerate(zip(front, side), 1):
            if f is None or s is None:
                return {"rejected": True, "reason": f"第 {i} 帧未识别到针尖"}
        arrays = [np.asarray(front, dtype=float), np.asarray(side, dtype=float)]
        maps = [np.asarray(front_map, dtype=float), np.asarray(side_map, dtype=float)]
        if any(a.shape != (len(front), 2) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("帧坐标必须是有限的二维坐标")
        if any(m.shape != (2, 2) or not np.isfinite(m).all() for m in maps):
            raise ValueError("映射矩阵必须是有限的 2×2 矩阵")
        for name, a in zip(("正面", "侧面"), arrays):
            spread = float(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2).max())
            if spread > self.stable_px:
                return {"rejected": True, "reason": f"工具未停稳（{name} 极差 {spread:.2f} px），请稍后再按"}
        metadata = _json_value(meta)
        means = [a.mean(axis=0) for a in arrays]
        mapped = [m @ mean for m, mean in zip(maps, means)]
        position = np.array([mapped[0][0], mapped[1][0], (mapped[0][1] + mapped[1][1]) / 2]) * 1000
        origin = np.asarray(self._presses[0]["position_um"]) if self._presses else position
        relative = position - origin
        noise = [float(np.sqrt(np.mean(np.sum(((a - mean) @ m.T * 1000) ** 2, axis=1))))
                 for a, mean, m in zip(arrays, means, maps)]
        entry = {"front_uv": arrays[0].tolist(), "side_uv": arrays[1].tolist(),
                 "front_mean_uv": means[0].tolist(), "side_mean_uv": means[1].tolist(),
                 "front_map": maps[0].tolist(), "side_map": maps[1].tolist(), "meta": metadata,
                 "position_um": position.tolist(), "relative_um": relative.tolist(),
                 "noise_um": {"front": noise[0], "side": noise[1]}}
        entries = self._presses + [entry]
        points = np.array([p["relative_um"] for p in entries])
        distances = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        maximum = float(distances[i, j])
        history = [p["result"]["max_deviation_um"] for p in self._presses] + [maximum]
        increments = np.diff(history)
        stable = (len(entries) >= 4 and len(increments) >= self.stable_presses and
                  bool(np.all(increments[-self.stable_presses:] < self.stable_delta_um)))
        passed = maximum <= self.max_mm * 1000
        diameter = _circle_diameter(points[:, :2])
        avg_noise = {key: float(np.mean([p["noise_um"][key] for p in entries])) for key in ("front", "side")}
        result = {"rejected": False, "count": len(entries), "delta_x_um": float(relative[0]),
                  "delta_y_um": float(relative[1]), "max_deviation_um": maximum,
                  "max_pair": [int(i + 1), int(j + 1)],
                  "range_um": dict(zip(("x", "y", "z"), np.ptp(points, axis=0).tolist())),
                  "circle_fit_diameter_um": diameter,
                  "circle_fit_reason": None if diameter is not None else "点数或角度覆盖不足",
                  "noise_um": avg_noise, "pass": passed, "stable": stable,
                  "message": f"第 {len(entries)} 次：最大偏差 {maximum:.1f} µm，{'合格' if passed else '不合格'}，{'已稳定' if stable else '未稳定'}"}
        entry["result"] = result
        self._presses.append(entry)
        return copy.deepcopy(result)

    def to_record(self):
        return copy.deepcopy({"max_mm": self.max_mm, "stable_px": self.stable_px,
                              "stable_presses": self.stable_presses,
                              "stable_delta_um": self.stable_delta_um,
                              "presses": self._presses})

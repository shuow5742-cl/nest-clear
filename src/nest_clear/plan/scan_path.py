"""按现场角点生成与一代一致的蛇形扫描路径。"""
import math

from nest_clear.config import Scan
from nest_clear.guard import GuardError


def guard_position(position: tuple[float, ...], travel: tuple[tuple[float, ...], ...], where: str) -> None:
    """检查三轴有限值和行程，供扫描及挑取下发前共用。"""
    if len(position) != 3 or len(travel) != 3 or any(not math.isfinite(v) or not bounds[0] <= v <= bounds[1] for v, bounds in zip(position, travel)):
        raise GuardError("POSITION-TRAVEL", f"坐标 {position} 超出行程 {travel} 或不是有限数", where, "检查扫描范围、工具补偿和设备行程。")


def _axis_points(start: float, end: float, step: float) -> list[float]:
    """按步距取点，末段不足一步时吸附到边界，与一代保持一致。"""
    if not all(math.isfinite(v) for v in (start, end, step)) or step <= 0:
        raise GuardError("SCAN-STEP", f"起点、终点、步距无效：{start}, {end}, {step}", "扫描路径", "检查角点和正步距。")
    if abs(start - end) < 1e-9:
        return [start]
    direction = 1.0 if end > start else -1.0
    points = [start]
    current = start
    while True:
        nxt = current + direction * step
        if (direction > 0 and nxt >= end) or (direction < 0 and nxt <= end):
            if abs(points[-1] - end) > 1e-9:
                points.append(end)
            return points
        points.append(nxt)
        current = nxt


def build_scan_path(scan: Scan, travel: tuple[tuple[float, ...], ...]) -> list[tuple[float, float, float]]:
    """由起点、X 终点、Y 终点构建路径，并守卫每个下发坐标。"""
    start, x_end, y_end = scan.corners
    if not math.isclose(start[1], x_end[1], abs_tol=1e-9) or not math.isclose(start[0], y_end[0], abs_tol=1e-9):
        raise GuardError("SCAN-CORNERS", f"三个角点不构成轴对齐矩形：{scan.corners}", "扫描路径", "按起点、同 Y 的 X 终点、同 X 的 Y 终点填写。")
    xs = _axis_points(start[0], x_end[0], scan.step_x)
    ys = _axis_points(start[1], y_end[1], scan.step_y)
    path = []
    for row, x in enumerate(xs):
        # 反转同一组点，避免边界吸附导致相邻行网格错位。
        for y in ys if row % 2 == 0 else reversed(ys):
            position = (x, y, scan.z)
            guard_position(position, travel, f"扫描位 {len(path)}")
            path.append(position)
    return path

"""按现场角点生成与一代一致的蛇形扫描路径。"""
import math

from nest_clear.config import Scan
from nest_clear.guard import GuardError, format_pose


def guard_position(position: tuple[float, ...], travel: tuple[tuple[float, ...], ...], where: str) -> None:
    """检查三轴有限值和行程，供扫描及挑取下发前共用。"""
    if len(position) != 3 or len(travel) != 3 or any(not math.isfinite(v) or not bounds[0] <= v <= bounds[1] for v, bounds in zip(position, travel)):
        raise GuardError("POSITION-TRAVEL", f"坐标 {format_pose(position)} 超出行程 {travel} 或不是有限数", where, "检查扫描范围、工具补偿和设备行程。")


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
    if scan.mode == "tray":
        return tray_path(scan, travel)
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


def tray_path(scan: Scan, travel: tuple[tuple[float, ...], ...]) -> list[tuple[float, float, float]]:
    """依据覆盖尺寸均匀分布中心，沿 Y 首行、沿 X 换行。"""
    if scan.fov_mm is None:
        raise GuardError("SCAN-FOV", "缺少视野尺寸", "scan.fov_mm", "填写视野或通过配置加载推算。")
    if scan.x_direction not in (-1, 1) or scan.y_direction not in (-1, 1):
        raise GuardError("SCAN-DIRECTION", "扫描方向无效", "scan.tray", "方向应为 1 或 -1。")
    axes = []
    for axis, origin, size, fov, direction in zip("xy", scan.origin_xy, scan.size_mm, scan.fov_mm, (scan.x_direction, scan.y_direction)):
        overlap = scan.min_overlap_mm
        if not all(math.isfinite(v) for v in (origin, size, fov, overlap)) or size <= 0 or overlap < 0 or fov <= overlap:
            raise GuardError("SCAN-TRAY", f"{axis} 轴尺寸={size}、视野={fov}、重叠={overlap} 无效", "scan.tray / scan.fov_mm", "尺寸须为正，重叠非负且小于视野。")
        n = 1 if size <= fov else math.ceil((size - overlap) / (fov - overlap))
        points = [origin + direction * (size / 2)] if n == 1 else [origin + direction * (fov / 2 + i * (size - fov) / (n - 1)) for i in range(n)]
        axes.append(points)
    path = []
    for row, x in enumerate(axes[0]):
        for y in axes[1] if row % 2 == 0 else reversed(axes[1]):
            position = (x, y, scan.z)
            guard_position(position, travel, f"扫描位 {len(path)}")
            path.append(position)
    return path


def scan_details(scan: Scan, path: list[tuple[float, float, float]]) -> dict:
    """生成启动日志和事件共用的网格摘要。"""
    xs = sorted({p[0] for p in path})
    ys = sorted({p[1] for p in path})
    overlaps = [None if len(axis) < 2 or scan.fov_mm is None else fov - (axis[-1] - axis[0]) / (len(axis) - 1) for axis, fov in zip((xs, ys), scan.fov_mm or (0, 0))]
    return {"mode": scan.mode, "fov_mm": scan.fov_mm, "nx": len(xs), "ny": len(ys), "overlap_mm": overlaps, "centers": path}

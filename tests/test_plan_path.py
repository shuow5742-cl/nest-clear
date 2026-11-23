"""验证扫描网格的边界吸附、反向行和行程守卫。"""
import pytest

from nest_clear.config import Scan
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import build_scan_path


def test_boundary_and_reverse() -> None:
    """末段不足步距时吸附边界，奇数行使用相同网格。"""
    scan = Scan(((3, 0), (0, 0), (3, 5)), 2, 3, 7)
    assert build_scan_path(scan, ((-10, 10), (-10, 10), (-10, 10))) == [(3, 0, 7), (3, 3, 7), (3, 5, 7), (1, 5, 7), (1, 3, 7), (1, 0, 7), (0, 0, 7), (0, 3, 7), (0, 5, 7)]


def test_travel_guard() -> None:
    """任何路径点越界都会在规划阶段失败。"""
    with pytest.raises(GuardError):
        build_scan_path(Scan(((3, 0), (0, 0), (3, 5)), 2, 3, 7), ((0, 2), (0, 5), (0, 8)))


@pytest.mark.parametrize(("scan", "expected"), [
    (Scan(((3, 0), (0, 0), (3, 5)), 2, 3, 4), [(3, 0), (3, 3), (3, 5), (1, 5), (1, 3), (1, 0), (0, 0), (0, 3), (0, 5)]),
    (Scan(((2, 4), (0, 4), (2, 0)), 1, 3, 4), [(2, 4), (2, 1), (2, 0), (1, 0), (1, 1), (1, 4), (0, 4), (0, 1), (0, 0)]),
    (Scan(((1, 2), (1, 2), (1, 2)), 1, 1, 4), [(1, 2)]),
])
def test_first_generation_expected(scan: Scan, expected: list[tuple[float, float]]) -> None:
    """期望点列来自一代 flow.py 的 build_serpentine_path 及 _axis_points。"""
    assert [point[:2] for point in build_scan_path(scan, ((-10, 10), (-10, 10), (-10, 10)))] == expected


def test_tray_grid_and_single():
    """默认二代视野覆盖一代料盘，并验证蛇形和重叠。"""
    from dataclasses import replace
    from nest_clear.config import load_config
    from nest_clear.plan.scan_path import scan_details
    cfg = load_config("config/nest_clear.yaml")
    cfg = replace(cfg, scan=replace(cfg.scan, mode="tray"), travel=((-500., 500.),)*3)
    path = build_scan_path(cfg.scan, cfg.travel)
    detail = scan_details(cfg.scan, path)
    assert (detail["nx"], detail["ny"]) == (3, 3)
    assert len(path) == 9
    assert all(overlap >= cfg.scan.min_overlap_mm for overlap in detail["overlap_mm"])
    small = replace(cfg.scan, size_mm=(10, 10))
    assert build_scan_path(small, cfg.travel) == [(-5, 5, small.z)]


@pytest.mark.parametrize("changes", [{"size_mm": (0, 10)}, {"fov_mm": (20, 84)}, {"min_overlap_mm": -1}])
def test_tray_guard(changes):
    """无效覆盖尺寸或重叠必须报告守卫错误。"""
    from dataclasses import replace
    from nest_clear.config import load_config
    cfg = load_config("config/nest_clear.yaml")
    with pytest.raises(GuardError):
        build_scan_path(replace(cfg.scan, mode="tray", **changes), ((-500., 500.),)*3)

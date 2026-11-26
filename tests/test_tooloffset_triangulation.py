"""主相机标定尖检测与多视角三角定位测试（T86-2）。"""
from dataclasses import replace
import math
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker import AsyncRequestWorker
from autoweaver.worker.clock import BTClock

from nest_clear.config import load_config
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.modes import Mode
from nest_clear.record.events import EventRecorder
from nest_clear.tooloffset.commands import send
from nest_clear.tooloffset.ref_tip_top import (
    RefTipTopConfig,
    RefTipTopDetector,
    render_ref_tip_top_overlay,
)
from nest_clear.tooloffset.triangulation import (
    TriangulationConfig,
    TriangulationSession,
    project_point,
    triangulate_rays,
    verify_target_xy_consistency,
)
from nest_clear.tooltip.store import initialize_tools
from nest_clear.vision.geometry import Calibration
from nest_clear.workers.tooloffset import ToolOffsetWorker


def _oblique_calibration() -> tuple[Calibration, tuple[int, int]]:
    """构造含 ~0.6° 倾斜与镜头畸变的相机标定参数（源自 test_camcal_core 假数据）。"""
    # Rodrigues([0.006, 0.0086, 0.02]) 的 Z 轴倾角约 0.60°
    K = np.array([[4650.0, 0.0, 2020.0], [0.0, 4655.0, 1510.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.06, 0.03, 0.0002, -0.0001, 0.0])
    R = cv2.Rodrigues(np.array([0.006, 0.0086, 0.02]))[0] @ np.diag([1.0, -1.0, -1.0])
    tilt_deg = float(np.degrees(np.arccos(abs(R[2, 2]))))
    assert 0.55 <= tilt_deg <= 0.65
    T = np.array([-52.5, -40.0, 0.0])
    cal = Calibration(K, D, R, T, 0.0)
    return cal, (4024, 3036)


def _render_projected_cone(
    pose: tuple[float, float, float],
    tip_xyz: tuple[float, float, float],
    cal: Calibration,
    size: tuple[int, int],
    *,
    bg_level: float = 225.0,
    cone_level: float = 45.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """用物理 3D 锥体网格经主相机真实斜视投影合成灰度图。"""
    w, h = size
    # 构造锥尖在 tip_xyz、竖直朝下的真实 3D 细长标定锥（高度 20mm，底面半径 1.2mm）
    tx, ty, tz = tip_xyz
    apex = np.array([tx, ty, tz], dtype=float)
    base_center = np.array([tx, ty, tz - 20.0], dtype=float)
    axis = base_center - apex
    axis_u = axis / np.linalg.norm(axis)
    ref_v = np.array([0.0, 0.0, 1.0]) if abs(axis_u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis_u, ref_v)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_u, e1)

    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    base_pts = [base_center + 1.2 * (math.cos(a) * e1 + math.sin(a) * e2) for a in angles]
    uv_apex = project_point(apex, pose, cal)
    assert uv_apex is not None
    uv_base = [project_point(p, pose, cal) for p in base_pts]
    assert all(p is not None for p in uv_base)

    # 先用凸包轮廓渲染，再反锯齿柔和混合；注意 INTER_AREA 降采样时高分辨中心需偏移 0.5*(ss-1)
    pts_2d = np.array([uv_apex, *[p for p in uv_base if p is not None]], dtype=np.float32)
    hull_idx = cv2.convexHull(pts_2d, returnPoints=False).ravel()
    poly = pts_2d[hull_idx]

    ss = 4
    shift = 0.5 * (ss - 1)
    hi = np.full((h * ss, w * ss), bg_level, dtype=np.float32)
    cv2.fillPoly(hi, [np.round(poly * ss + shift).astype(np.int32)], float(cone_level), lineType=cv2.LINE_AA)
    img = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    if noise_sigma > 0 and rng is not None:
        img = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _draw_2d_cone(
    size: tuple[int, int],
    apex: tuple[float, float],
    angle_deg: float,
    half_angle_deg: float = 24.0,
    length: float = 95.0,
    *,
    bg: str = "white",
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.8,
    seed: int = 0,
) -> np.ndarray:
    """绘制任意朝向 2D 合成锥尖图（支持白/灰/渐变背景）。"""
    w, h = size
    if bg == "white":
        base = np.full((h, w), 238.0, dtype=np.float32)
    elif bg == "gray":
        base = np.full((h, w), 165.0, dtype=np.float32)
    else:
        gx = np.linspace(150.0, 230.0, w, dtype=np.float32)[None, :]
        gy = np.linspace(-10.0, 10.0, h, dtype=np.float32)[:, None]
        base = gx + gy

    rad = math.radians(angle_deg)
    ha = math.radians(half_angle_deg)
    ax, ay = apex
    # 锥体从 apex 沿 angle_deg 反方向张开
    p1 = (ax - length * math.cos(rad - ha), ay - length * math.sin(rad - ha))
    p2 = (ax - length * math.cos(rad + ha), ay - length * math.sin(rad + ha))
    poly = np.array([apex, p1, p2], dtype=np.float32)

    ss = 4
    shift = 0.5 * (ss - 1)
    mask = np.zeros((h * ss, w * ss), dtype=np.float32)
    cv2.fillPoly(mask, [np.round(poly * ss + shift).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    img = base * (1.0 - 0.72 * mask)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("angle_deg", "bg", "noise_sigma"),
    [
        (42.0, "white", 0.0),
        (135.0, "gray", 1.5),
        (-65.0, "gradient", 2.0),
        (215.0, "white", 1.0),
    ],
)
def test_ref_tip_top_detector_synthetic_cones(angle_deg, bg, noise_sigma):
    apex_true = (432.35, 318.65)
    img = _draw_2d_cone((960, 720), apex_true, angle_deg, bg=bg, noise_sigma=noise_sigma, seed=7)
    det = RefTipTopDetector()
    res = det.detect(img)
    assert res.point_uv is not None, res.diagnostic
    assert res.focused is True
    assert res.quality > 0.4
    err_px = math.hypot(res.point_uv[0] - apex_true[0], res.point_uv[1] - apex_true[1])
    assert err_px < 0.85

    ovl = render_ref_tip_top_overlay(img, res, shot_index=1, principal=(480.0, 360.0))
    assert ovl.shape == (720, 960, 3)


def test_ref_tip_top_detector_missing_and_non_unique():
    det = RefTipTopDetector()
    blank = np.full((640, 480), 230, dtype=np.uint8)
    r_miss = det.detect(blank)
    assert r_miss.point_uv is None
    assert "未识别到标定尖" in r_miss.diagnostic

    # 两个清晰锥体 -> 不唯一报错
    img1 = _draw_2d_cone((960, 720), (300.0, 260.0), 45.0, noise_sigma=0.0)
    img2 = _draw_2d_cone((960, 720), (680.0, 460.0), 120.0, noise_sigma=0.0)
    both = np.minimum(img1, img2)
    r_dup = det.detect(both)
    assert r_dup.point_uv is None
    assert "不唯一" in r_dup.diagnostic


def test_triangulation_noiseless_and_noisy_accuracy_with_06deg_tilt():
    cal, image_size = _oblique_calibration()
    # 主相机相对模组有 (-52.5, -40.0) 偏移且到不了标定尖正上方：
    # 标定尖实际在 (92.0, 30.0, 12.0)，相机在 (162.0, 70.0, 122.0) 时光心在 (109.5, 30.0, 122.0)，处于 X+ 一侧斜看
    tip_true = (92.0, 30.0, 12.0)
    first_pose = (162.0, 70.0, 122.0)
    travel = ((162.0, 260.0), (0.0, 160.0), (80.0, 150.0))
    tcfg = TriangulationConfig(
        frames_per_shot=3,
        baseline_min_mm=20.0,
        baseline_max_mm=30.0,
        min_views=3,
        max_views=5,
        max_ray_residual_mm=0.02,
        min_baseline_ratio=0.15,
        approx_tip_z_mm=12.0,
    )

    # 1) 无噪声精确投影三角定位：XY 误差 <= 0.002 mm
    sess = TriangulationSession(cal, travel, image_size, tcfg)
    pose = first_pose
    while True:
        uv = project_point(tip_true, pose, cal)
        assert uv is not None
        status = sess.add_observation(pose, uv)
        if status != "move":
            break
        pose = sess.target_pose
        assert pose is not None
        assert pose[0] >= first_pose[0] - 1e-9
        bl = math.hypot(pose[0] - first_pose[0], pose[1] - first_pose[1])
        assert 20.0 - 1e-6 <= bl <= 30.0 + 1e-6
        assert travel[0][0] <= pose[0] <= travel[0][1]
        assert travel[1][0] <= pose[1] <= travel[1][1]
        pred = project_point(tip_true, pose, cal)
        assert pred is not None
        assert 0.1 * image_size[0] <= pred[0] <= 0.9 * image_size[0]
        assert 0.1 * image_size[1] <= pred[1] <= 0.9 * image_size[1]

    assert status == "done"
    assert sess.result is not None
    err_xy_noiseless = math.hypot(sess.result.tip_xyz_mm[0] - tip_true[0], sess.result.tip_xyz_mm[1] - tip_true[1])
    err_z_noiseless = abs(sess.result.tip_xyz_mm[2] - tip_true[2])
    assert err_xy_noiseless <= 0.002
    assert err_z_noiseless <= 0.005
    # 验证与生产 pixel_to_module 完全一致
    diff_prod = verify_target_xy_consistency(sess.result.tip_xyz_mm, first_pose, project_point(tip_true, first_pose, cal), cal)
    assert diff_prod < 1e-6

    # 2) 带合成渲染与检测+三帧中位+像素噪声三角定位：XY 误差 <= 0.01 mm
    # （用真实像素重投影 + 0.12 px 噪声 3 帧中位，同时用真实 RefTipTopDetector 验证合成图像）
    det = RefTipTopDetector()
    rng = np.random.default_rng(123)
    sess_noisy = TriangulationSession(cal, travel, image_size, tcfg)
    pose = first_pose
    while True:
        # 用 4024x3036 的 1/4 缩略验证 RefTipTopDetector，并用真 K 投影+检测抖动做 3 帧中位
        uv_true = project_point(tip_true, pose, cal)
        assert uv_true is not None
        frames_uv = [
            (uv_true[0] + float(rng.normal(0.0, 0.12)), uv_true[1] + float(rng.normal(0.0, 0.12)))
            for _ in range(3)
        ]
        med_uv = (
            float(np.median([p[0] for p in frames_uv])),
            float(np.median([p[1] for p in frames_uv])),
        )
        status = sess_noisy.add_observation(pose, med_uv)
        if status != "move":
            break
        pose = sess_noisy.target_pose
        assert pose is not None
    assert status == "done" and sess_noisy.result is not None
    err_xy_noisy = math.hypot(sess_noisy.result.tip_xyz_mm[0] - tip_true[0], sess_noisy.result.tip_xyz_mm[1] - tip_true[1])
    assert err_xy_noisy <= 0.01

    # 3) 真实渲染合成锥图端到端单目检测验证
    cal_small = Calibration(
        np.array([[1160.0, 0.0, 505.0], [0.0, 1162.0, 378.0], [0.0, 0.0, 1.0]]),
        cal.distortion,
        cal.rotation,
        cal.translation,
        0.0,
    )
    img = _render_projected_cone(first_pose, tip_true, cal_small, (1006, 759), noise_sigma=1.5, rng=rng)
    r_cone = det.detect(img)
    assert r_cone.point_uv is not None, r_cone.diagnostic
    uv_small_true = project_point(tip_true, first_pose, cal_small)
    assert math.hypot(r_cone.point_uv[0] - uv_small_true[0], r_cone.point_uv[1] - uv_small_true[1]) < 1.2


def test_triangulation_adaptive_4th_5th_view_and_failure():
    cal, image_size = _oblique_calibration()
    tip_true = (92.0, 30.0, 12.0)
    first_pose = (162.0, 70.0, 122.0)
    travel = ((162.0, 260.0), (0.0, 160.0), (80.0, 150.0))

    # 让第 3 个视角有较大射线偏差但第 4 个视角不增加、反而用残差阈值触发加拍，或者用过大误差导致 5 个视角后失败
    tcfg = TriangulationConfig(min_views=3, max_views=5, max_ray_residual_mm=0.02, min_baseline_ratio=0.15, approx_tip_z_mm=12.0)
    sess = TriangulationSession(cal, travel, image_size, tcfg)

    uv1 = project_point(tip_true, first_pose, cal)
    assert sess.add_observation(first_pose, uv1) == "move"
    p2 = sess.target_pose
    uv2 = project_point(tip_true, p2, cal)
    assert sess.add_observation(p2, uv2) == "move"
    p3 = sess.target_pose
    uv3 = project_point(tip_true, p3, cal)
    # 故意给第 3 个点加 12 px 偏差 -> max_residual 远超 0.02mm -> 应加拍第 4、第 5 个视角并在第 5 次仍失败
    assert sess.add_observation(p3, (uv3[0] + 12.0, uv3[1] - 9.0)) == "move"
    p4 = sess.target_pose
    assert p4 is not None
    assert sess.add_observation(p4, project_point(tip_true, p4, cal)) == "move"
    p5 = sess.target_pose
    assert p5 is not None
    assert sess.add_observation(p5, project_point(tip_true, p5, cal)) == "failed"
    assert sess.failure is not None
    assert "5 个视角" in sess.failure


class _SimPLC(AsyncRequestWorker):
    name = "plc"

    def __init__(self):
        super().__init__()
        self.request = (0, 0, None)
        self.actual = (162.0, 70.0, 122.0)
        self.replies = {}

    def execute(self, payload):
        op = payload["operation"]
        if op == "request":
            return self.request
        if op == "actual":
            return self.actual
        if op == "calibration_final":
            return self.actual
        if op == "reply":
            self.replies[payload["task_id"]] = (payload["func"], payload["coordinates"])


class _SimMainCamera(AsyncRequestWorker):
    name = "camera"

    def __init__(self, plc: _SimPLC, cal: Calibration, tip_xyz: tuple[float, float, float], size: tuple[int, int]):
        super().__init__()
        self.plc = plc
        self.cal = cal
        self.tip_xyz = tip_xyz
        self.size = size
        self.blank = False
        self.rng = np.random.default_rng(42)

    def execute(self, payload):
        if self.blank:
            return np.full((self.size[1], self.size[0]), 230, dtype=np.uint8), {}
        return _render_projected_cone(self.plc.actual, self.tip_xyz, self.cal, self.size, noise_sigma=0.8, rng=self.rng), {}


def test_bt_90_triangulation_loop_to_91_and_error_99(tmp_path):
    cal, _ = _oblique_calibration()
    # 使用快速尺寸相机（内参按 1/4 缩放保持视场一致）跑完整 BT 状态机与图像保存
    cal_bt = Calibration(
        np.array([[1162.5, 0.0, 505.0], [0.0, 1163.75, 377.5], [0.0, 0.0, 1.0]]),
        cal.distortion,
        cal.rotation,
        cal.translation,
        0.0,
    )
    img_size = (1006, 759)
    tip_true = (92.0, 30.0, 12.0)

    cfg = load_config("config/nest_clear.yaml")
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    initialize_tools(cal_dir / "tools.yaml")
    cam_path = cal_dir / "camera.yaml"
    cam_path.write_text("test", encoding="utf-8")
    cfg = replace(
        cfg,
        tools=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        tool_calibration=replace(cfg.tool_calibration, directory=str(cal_dir)),
        travel=((160.0, 260.0), (0.0, 160.0), (80.0, 150.0)),
        tool_offset={
            **cfg.tool_offset,
            "settle_window_s": 0.001,
            "triangulation": {
                **cfg.tool_offset["triangulation"],
                "approx_tip_z_mm": 12.0,
                "max_ray_residual_mm": 0.15,
            },
        },
    )

    events = EventRecorder(tmp_path, run=True)
    board = WorldBoard()
    plc = _SimPLC()
    plc.actual = (162.0, 70.0, 122.0)
    cam = _SimMainCamera(plc, cal_bt, tip_true, img_size)
    clock = BTClock(board)
    for w in (plc, cam, ToolOffsetWorker()):
        clock.attach_worker(w)

    batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal_bt, calibration_path=cam_path)
    handle = clock.submit(batch)

    def until(pred, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            clock.tick_once()
            assert handle.result is None
            if pred():
                return
            time.sleep(0.001)
        raise AssertionError("等待超时")

    def request(task_id: int, code: int):
        plc.request = (task_id, code, 1)
        until(lambda: task_id in plc.replies)
        return plc.replies[task_id]

    try:
        task = 1
        poses_seen = []
        while True:
            poses_seen.append(plc.actual)
            func, coords = request(task, 90)
            assert coords["pc_tool_id"] == 1
            if func == 91:
                break
            assert func == 90
            nxt = coords["scan_xyz"]
            assert nxt[0] >= 162.0 - 1e-9
            plc.actual = tuple(float(v) for v in nxt)
            task += 1

        assert len(poses_seen) >= 3
        session = batch.blackboard.read("tooloffset_session")
        state = session["triangulation_state"]
        assert len(state["views"]) == len(poses_seen)
        assert state["xyz"] is not None
        assert len(state["target_xy"]) == 2
        assert session["phase"] == "wait_tool_start"
        assert (tmp_path / "live" / "tooloffset_tri_1.jpg").is_file()

        # 再发一次遮挡无尖的 90 会话测试错误回 99
        send(tmp_path, "tooloffset_abort")
        task += 1
        func_abort, coords_abort = request(task, 100)
        assert (func_abort, coords_abort) == (99, {"pc_tool_id": 1})

        cam.blank = True
        plc.actual = (162.0, 70.0, 122.0)
        task += 1
        func_err, coords_err = request(task, 90)
        assert (func_err, coords_err) == (99, {"pc_tool_id": 1})
        assert batch.blackboard.read("mode") == Mode.IDLE
    finally:
        clock.shutdown()

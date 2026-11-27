"""160 下针深度标定 ChArUco 标定板测量面单元测试与 TCP 端到端测试。"""

from dataclasses import replace
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from autoweaver.motion_policy.world_board import WorldBoard
from autoweaver.worker import AsyncRequestWorker
from autoweaver.worker.clock import BTClock

from nest_clear.camcal.session import _board
from nest_clear.comm import Handshake, PLCClient
from nest_clear.config import load_config
from nest_clear.depthcal.layout import (
    charuco_board_outer_quad, charuco_corner_id, charuco_safe_quad, point_in_board_safe_zone,
)
from nest_clear.depthcal.record import DepthRecord
from nest_clear.depthcal.session import Session
from nest_clear.flow.batch import build_batch
from nest_clear.flow.context import RequestTracker
from nest_clear.flow.modes import Mode
from nest_clear.guard import GuardError
from nest_clear.paths import live_image_file
from nest_clear.record.events import EventRecorder
from nest_clear.vision.geometry import Calibration
from nest_clear.workers.plc import PlcWorker
from sim.charuco_scene import (
    BOARD_ORIGIN, BOARD_ROTATION, DISTORTION, K, ROTATION, CharucoScene,
)
from sim.fake_plc import FakePLC


def _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.01):
    """构造位于模组正坐标、带微倾斜的 ChArUco 场景及对应主相机标定。"""
    plane_rot = cv2.Rodrigues(np.array([0.002, 0.0028, yaw_rad]))[0]
    board_rot = plane_rot @ np.diag([1.0, -1.0, -1.0])
    board_origin = (
        np.array([-52.5, -40.0, 0.0]) + plane_rot @ np.array([0.0, 80.0, 0.0]) + np.array(shift, dtype=float)
    )
    scene = CharucoScene(
        k=K,
        distortion=DISTORTION,
        rotation=ROTATION,
        board_rotation=board_rot,
        board_origin=board_origin,
    )
    cal = Calibration(
        K, DISTORTION, ROTATION, np.zeros(3), float(board_origin[2]),
        False, board_origin, board_rot[:, 2],
    )
    return scene, cal, board_origin, board_rot


def _film_patch_image(size=(420, 420), target_id=2):
    """生成单张带 V2 菲林圆环+ArUco 的合成图。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, target_id, 80), cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = np.full((size[1], size[0]), 195, np.uint8)
    u, v = size[0] // 2, size[1] // 2
    cv2.circle(img, (u, v), 37, 20, 6, cv2.LINE_AA)
    cv2.line(img, (u - 90, v), (u + 90, v), 20, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - 90), (u, v + 90), 20, 2, cv2.LINE_AA)
    img[v - 180:v - 100, u - 180:u - 100] = np.where(marker < 128, 20, 195).astype(np.uint8)
    return img


def test_depth_target_config_validation(tmp_path):
    """depth_calibration.target 默认为 auto，board_keepout_mm 默认 5.0 且校验 0 ≤ 值 ≤ 2×square_mm−2。"""
    cfg = load_config('config/nest_clear.yaml')
    assert cfg.depth_calibration['target'] == 'auto'
    assert cfg.depth_calibration['board_keepout_mm'] == 5.0

    raw = yaml.safe_load(Path('config/nest_clear.yaml').read_text(encoding='utf-8'))
    for val in ('auto', 'film', 'charuco'):
        good = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'target': val}}
        p = tmp_path / f'{val}.yaml'
        p.write_text(yaml.safe_dump(good), encoding='utf-8')
        assert load_config(p, require_tools=False).depth_calibration['target'] == val

    for keepout in (0, 0.0, 5.0, 8.0):
        good_k = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'board_keepout_mm': keepout}}
        pk = tmp_path / f'k_{keepout}.yaml'
        pk.write_text(yaml.safe_dump(good_k), encoding='utf-8')
        assert load_config(pk, require_tools=False).depth_calibration['board_keepout_mm'] == float(keepout)

    for bad_k in (-0.1, -5, 8.1, 10, 10.0, True, 'ten'):
        bad_doc = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'board_keepout_mm': bad_k}}
        pbk = tmp_path / 'bad_k.yaml'
        pbk.write_text(yaml.safe_dump(bad_doc), encoding='utf-8')
        with pytest.raises(GuardError, match='depth_calibration.board_keepout_mm'):
            load_config(pbk, require_tools=False)

    # square_mm 增大时上限按 2×square_mm−2 联动放宽
    sq6_ok = {
        **raw,
        'camera_calibration': {**raw['camera_calibration'], 'square_mm': 6.0},
        'depth_calibration': {**raw['depth_calibration'], 'board_keepout_mm': 10.0},
    }
    p_sq6 = tmp_path / 'sq6_ok.yaml'
    p_sq6.write_text(yaml.safe_dump(sq6_ok), encoding='utf-8')
    assert load_config(p_sq6, require_tools=False).depth_calibration['board_keepout_mm'] == 10.0
    sq6_bad = {**sq6_ok, 'depth_calibration': {**sq6_ok['depth_calibration'], 'board_keepout_mm': 10.1}}
    p_sq6_bad = tmp_path / 'sq6_bad.yaml'
    p_sq6_bad.write_text(yaml.safe_dump(sq6_bad), encoding='utf-8')
    with pytest.raises(GuardError, match='depth_calibration.board_keepout_mm'):
        load_config(p_sq6_bad, require_tools=False)

    bad = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'target': 'chessboard'}}
    p_bad = tmp_path / 'bad.yaml'
    p_bad.write_text(yaml.safe_dump(bad), encoding='utf-8')
    with pytest.raises(GuardError, match='depth_calibration.target'):
        load_config(p_bad, require_tools=False)


def test_charuco_safe_zone_geometry():
    """板外轮廓内缩 board_keepout_mm 与允许区包含判定在正反旋向下均正确。"""
    camcal_cfg = {'cols': 21, 'rows': 16, 'square_mm': 5.0}
    # 含反射与平移的刚体变换（cols*square=105, rows*square=80）
    quad = charuco_board_outer_quad(camcal_cfg, lambda pts: np.asarray(pts, float) * [1.0, -1.0] + [40.0, 130.0])
    assert quad.shape == (4, 2)
    safe = charuco_safe_quad(quad, 10.0)
    assert safe is not None
    assert point_in_board_safe_zone((50.0, 120.0), quad, 10.0)
    assert point_in_board_safe_zone((135.0, 60.0), quad, 10.0)
    assert not point_in_board_safe_zone((48.5, 120.0), quad, 10.0)
    assert not point_in_board_safe_zone((92.5, 125.0), quad, 10.0)
    assert point_in_board_safe_zone((40.0, 130.0), quad, 0.0)
    assert not point_in_board_safe_zone((39.5, 130.0), quad, 0.0)
    assert charuco_safe_quad(quad, 45.0) is None
    assert not point_in_board_safe_zone((92.5, 90.0), quad, 45.0)


def test_charuco_scene_partial_view_pose_plan_center_and_roi(tmp_path):
    """主相机局部可见 ChArUco 板时：二维位姿拟合、5 测点规划（含标签与模组坐标）、居中迭代收敛、ROI 正确。"""
    scene, cal, board_origin, board_rot = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.015)
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, depth_calibration={
        **cfg.depth_calibration,
        'target': 'auto',
        'camera_z_min_mm': -12.0,
        'center_tol_px': 15.0,
        'roi_mm': 8.0,
        'travel': ((30.0, 170.0), (20.0, 160.0), (-12.0, 0.0)),
    })
    events = EventRecorder(tmp_path, run=True)
    sess = Session(cfg, cal, DepthRecord(events))
    sess.record.begin()

    # 初始相机位于板偏右下局部视野（仅看见部分角点，且未居中在计划角点上）
    start_pose = (118.0, 74.0, -3.0)
    sess.actual = start_pose
    sess.arrived(start_pose)
    partial_img = scene.render(start_pose, focus_z=-3.0)
    assert partial_img.shape == (3036, 4024)

    next_xyz = sess.advance([partial_img])
    assert sess.target_type == 'charuco'
    assert sess.target_type_label == 'ChArUco 标定板（厚 3.15mm）'
    assert sess.board_fit is not None
    assert sess.board_fit['corner_count'] >= 6
    assert sess.board_fit['rms_residual_mm'] < 0.02

    # 验证 5 个主测点（四角内缩 2 格 + 中心）及 4 个加测边中点
    assert len(sess.plan) == 5
    assert set(sess.point_positions.values()) >= {'左上', '右上', '右下', '左下', '中心', '上中', '右中', '下中', '左中'}
    assert set(sess.plan) == {
        k for k, v in sess.point_positions.items() if v in {'左上', '右上', '右下', '左下', '中心'}
    }
    assert set(sess.extra) == {
        k for k, v in sess.point_positions.items() if v in {'上中', '右中', '下中', '左中'}
    }
    assert sess.plan == ['C278', 'C149', 'C261', 'C038', 'C021'] or set(sess.plan) == {'C021', 'C038', 'C278', 'C261', 'C149'}

    # 验证规划角点模组坐标与真实板面三维投影一致（< 0.05 mm）
    all_board_pts = _board(scene.config).getChessboardCorners()
    for lbl in sess.plan:
        cid = charuco_corner_id(lbl)
        true_mod = (all_board_pts[cid] @ board_rot.T + board_origin)[:2]
        assert np.linalg.norm(sess.layout[lbl] - true_mod) < 0.05

    # 验证居中迭代收敛（人为从偏差 2mm 开始居中）
    assert next_xyz is not None and sess.phase == 'center'
    first_lbl = sess.label
    offset_pose = (next_xyz[0] + 2.0, next_xyz[1] - 1.5, next_xyz[2])
    sess.actual = offset_pose
    sess.arrived(offset_pose)
    refine_xyz = sess.advance([scene.render(offset_pose, focus_z=-3.0)])
    assert sess.phase == 'center' and sess.center_round == 2
    assert math.hypot(refine_xyz[0] - next_xyz[0], refine_xyz[1] - next_xyz[1]) < 0.1

    sess.actual = refine_xyz
    sess.arrived(refine_xyz)
    step_after_center = sess.advance([scene.render(refine_xyz, focus_z=-3.0)])
    assert step_after_center is not None
    assert sess.roi is not None
    y_sl, x_sl = sess.roi
    assert (y_sl.stop - y_sl.start) == (x_sl.stop - x_sl.start)
    # 4024x3036 主相机约 23.6 µm/px，8 mm ROI 约 340 px
    assert 300 <= (x_sl.stop - x_sl.start) <= 380
    roi_crop = scene.render(refine_xyz, focus_z=-3.0)[sess.roi]
    assert roi_crop.max() - roi_crop.min() > 100


def test_auto_mode_prefers_film_then_charuco_and_missing_raises_depth_target(tmp_path):
    """auto 模式先菲林后 ChArUco，会话内锁定；两者均无报 DEPTH-TARGET；显式模式不跨类型识别。"""
    scene, cal, _, _ = _shifted_scene(shift=(95.0, 88.0, -105.0))
    cfg = load_config('config/nest_clear.yaml')
    base_dc = {
        **cfg.depth_calibration,
        'camera_z_min_mm': -12.0,
        'travel': ((30.0, 170.0), (20.0, 160.0), (-12.0, 0.0)),
    }

    # 1) auto 模式：菲林图优先识别为菲林并锁定
    film_cal = Calibration(
        np.array([[2000., 0., 210.], [0., 2000., 210.], [0., 0., 1.]]),
        np.zeros(5), np.diag([1., -1., -1.]), np.zeros(3), -105.0,
    )
    s_film = Session(replace(cfg, depth_calibration={**base_dc, 'target': 'auto'}), film_cal, DepthRecord(EventRecorder(tmp_path / 'f', run=False)))
    s_film.actual = (95.0, 88.0, -3.0)
    s_film.arrived(s_film.actual)
    s_film.advance([_film_patch_image(target_id=2)])
    assert s_film.target_type == 'film'
    assert s_film.target_type_label == '菲林'
    # 锁定菲林后，后续即使给 ChArUco 图也不能切类型，报 DEPTH-TARGET
    with pytest.raises(GuardError) as exc_locked:
        s_film.center(scene.render((95.0, 88.0, -3.0), focus_z=-3.0))
    assert exc_locked.value.code == 'DEPTH-TARGET'

    # 2) auto 模式：无菲林但有 ChArUco -> 识别为 ChArUco
    s_char = Session(replace(cfg, depth_calibration={**base_dc, 'target': 'auto'}), cal, DepthRecord(EventRecorder(tmp_path / 'c', run=False)))
    s_char.actual = (95.0, 88.0, -3.0)
    s_char.arrived(s_char.actual)
    s_char.advance([scene.render((95.0, 88.0, -3.0), focus_z=-3.0)])
    assert s_char.target_type == 'charuco'
    assert s_char.target_type_label == 'ChArUco 标定板（厚 3.15mm）'

    # 3) auto 模式：空白图（两者都没有）-> DEPTH-TARGET 且写明“未识别到菲林靶点或标定板”
    s_blank = Session(replace(cfg, depth_calibration={**base_dc, 'target': 'auto'}), cal, DepthRecord(EventRecorder(tmp_path / 'b', run=False)))
    s_blank.actual = (95.0, 88.0, -3.0)
    s_blank.arrived(s_blank.actual)
    with pytest.raises(GuardError) as exc_blank:
        s_blank.advance([np.full((420, 420), 200, np.uint8)])
    assert exc_blank.value.code == 'DEPTH-TARGET'
    assert '未识别到菲林靶点或标定板' in exc_blank.value.what

    # 4) 显式 film 模式不认 ChArUco；显式 charuco 模式不认菲林
    s_only_film = Session(replace(cfg, depth_calibration={**base_dc, 'target': 'film'}), cal, DepthRecord(EventRecorder(tmp_path / 'of', run=False)))
    s_only_film.actual = (95.0, 88.0, -3.0)
    s_only_film.arrived(s_only_film.actual)
    with pytest.raises(GuardError) as exc_of:
        s_only_film.advance([scene.render((95.0, 88.0, -3.0), focus_z=-3.0)])
    assert exc_of.value.code == 'DEPTH-TARGET'

    s_only_char = Session(replace(cfg, depth_calibration={**base_dc, 'target': 'charuco'}), film_cal, DepthRecord(EventRecorder(tmp_path / 'oc', run=False)))
    s_only_char.actual = (95.0, 88.0, -3.0)
    s_only_char.arrived(s_only_char.actual)
    with pytest.raises(GuardError) as exc_oc:
        s_only_char.advance([_film_patch_image(target_id=2)])
    assert exc_oc.value.code == 'DEPTH-TARGET'


class SimCharucoDepthCamera(AsyncRequestWorker):
    """用于 160 端到端测试的模拟 ChArUco 相机（真实局部投影 + 连续 Z 向散焦模糊）。"""
    name = 'camera'

    def __init__(self, scene, board_origin, board_rot, *, start_xyz=(95.0, 88.0, -3.0)):
        super().__init__()
        self.scene = scene
        self.board_origin = board_origin
        self.board_rot = board_rot
        self.actual = tuple(start_xyz)
        self.exposure_us = 6000.0
        self.exposure_auto = 0
        self._sharp_cache = {}

    def _focus_z_for_xy(self, x, y):
        # 轻微平面倾斜：板面焦平面随 X/Y 线性变化，真实残差 0（≤ 0.02mm）
        return -5.020 + 0.0004 * (x - 95.0) - 0.0003 * (y - 88.0)

    def execute(self, payload):
        op = payload.get('operation')
        if op == 'get_exposure_state':
            return None, {'exposure_us': self.exposure_us, 'exposure_auto': self.exposure_auto}
        if op == 'restore_exposure_state':
            if payload.get('exposure_us') is not None:
                self.exposure_us = float(payload['exposure_us'])
            if payload.get('exposure_auto') is not None:
                self.exposure_auto = payload['exposure_auto']
            return None, {'exposure_us': self.exposure_us, 'exposure_auto': self.exposure_auto}
        if op == 'set_exposure':
            self.exposure_us = float(payload['exposure_us'])
            return None, {'exposure_us': self.exposure_us}

        x, y, z = self.actual
        key = (round(float(x), 3), round(float(y), 3))
        base = self._sharp_cache.get(key)
        if base is None:
            base = self.scene.render((x, y, -3.0), focus_z=-3.0)
            self._sharp_cache[key] = base
        z_focus = self._focus_z_for_xy(x, y)
        sigma = math.hypot(0.4, 1.5 * (z - z_focus))
        frame = cv2.GaussianBlur(base, (0, 0), sigma)
        return frame, {'exposure_us': self.exposure_us}


def test_charuco_e2e_tcp_160_to_161_and_out_of_travel_retest(tmp_path):
    """端到端 TCP：ChArUco 模式完整 160→161，一个计划角点不在行程内自动补测邻近内角点并写明板面平面。"""
    # yaw=0.05 rad 使右上角 C038 落在 Y≈120.1 mm（> travel_y_max 118.0 mm），其邻近内角点 C058（Y≈115.1）及其他 4 点均在行程内
    scene, cal, board_origin, board_rot = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.05)
    camera = SimCharucoDepthCamera(scene, board_origin, board_rot, start_xyz=(95.0, 88.0, -3.0))

    cfg = load_config('config/nest_clear.yaml')
    travel = ((32.0, 158.6), (-15.0, 118.0), (-12.0, 0.0))
    settings = {
        **cfg.depth_calibration,
        'target': 'auto',
        'camera_z_min_mm': -9.0,
        'travel': travel,
        'coarse_step_mm': 1.0,
        'overshoot_mm': 0.2,
        'repeats_per_point': 2,
        'frames_per_step': 1,
        'fine_steps_target': 14,
        'settle_window_s': 0.0001,
        'arrival_timeout_s': 0.05,
    }
    cfg = replace(
        cfg,
        depth_calibration=settings,
        travel=travel,
        camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 3.15},
    )
    events = EventRecorder(tmp_path, run=True)

    with FakePLC() as plc:
        io = PLCClient('127.0.0.1', plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        plc_worker = PlcWorker(Handshake(io, plc.contract, events=events))
        for worker in (plc_worker, camera):
            clock.attach_worker(worker)
        batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal)
        clock.submit(batch)

        def exchange(xyz=None):
            if xyz is not None:
                camera.actual = tuple(xyz)
            plc.set_actual_xyz(camera.actual)
            task = plc.request(160, tool_id=65535)
            deadline = time_monotonic() + 8.0
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                if time_monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 应答')
                time_sleep(0.0002)
            code = plc.io.read('reply_func')
            coords = plc.io.read_real32_block(40942, 3) if code == 160 else None
            return code, coords

        import time
        time_monotonic = time.monotonic
        time_sleep = time.sleep
        try:
            target = None
            session = None
            for _ in range(1200):
                code, target = exchange(target)
                if code == 161:
                    break
                assert code == 160, events.status.data.get('depthcal')
                session = batch.blackboard.read('depth_session')
                assert target is not None
                # 校验所有下发 XY 均落在 depth_calibration.travel 内
                assert travel[0][0] <= target[0] <= travel[0][1]
                assert travel[1][0] <= target[1] <= travel[1][1]
                assert travel[2][0] <= target[2] <= travel[2][1]
            else:
                pytest.fail('ChArUco 160 会话未完成')
        finally:
            clock.shutdown()
            io.close()

    assert batch.blackboard.read('mode') == Mode.IDLE
    assert session is not None
    assert session.target_type == 'charuco'
    assert session.target_type_label == 'ChArUco 标定板（厚 3.15mm）'
    assert len(session.results) == 5

    # 验证不在行程内的 C038 被邻近内角点（如 C058）成功补测
    assert len(session.failed_points) == 1
    fp = session.failed_points[0]
    assert fp['label'] == 'C038'
    assert fp['error_code'] == 'POSITION-TRAVEL'
    assert fp['retest_status'] == 'success'
    retest_lbl = fp['retest_label']
    assert retest_lbl and retest_lbl != 'C038'
    retest_res = next(r for r in session.results if r['label'] == retest_lbl)
    assert retest_res['retest_of'] == 'C038'

    # 验证落盘记录与界面状态写明测量面类型、板面平面与可行性结论
    session_dir = session.record.directory
    fits_doc = json.loads((session_dir / 'fits.json').read_text(encoding='utf-8'))
    plane_doc = json.loads((session_dir / 'plane.json').read_text(encoding='utf-8'))
    rec_doc = json.loads((session_dir / 'session_record.json').read_text(encoding='utf-8'))
    status_depth = events.status.data['depthcal']

    for doc in (fits_doc, rec_doc, status_depth):
        assert doc['target_type'] == 'charuco'
        assert doc['target_type_label'] == 'ChArUco 标定板（厚 3.15mm）'
        assert doc['board_fit']['corner_count'] >= 6
        assert doc['board_fit']['rms_residual_mm'] < 0.02
    for doc in (fits_doc, status_depth):
        assert doc['feasibility']['target_type'] == 'charuco'
        assert doc['feasibility']['plane_verdict'] == '可行'
    assert '板面平面（板厚 3.15mm，盘底 = 板面 − 板厚方向换算，暂不用于生产）' in plane_doc['final']['plane_note']
    assert status_depth['phase'] == 'done'
    assert live_image_file(tmp_path, 'depthcal_roi').is_file()


def test_charuco_safe_zone_rejects_out_of_board_plan_and_centering(tmp_path, monkeypatch):
    """模拟板位姿使计划点或居中目标落在安全区外：不下发该移动、记原因并换邻近安全区内测点。"""
    import nest_clear.depthcal.session as sess_mod

    scene, cal, _, _ = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.015)
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, depth_calibration={
        **cfg.depth_calibration,
        'target': 'charuco',
        'camera_z_min_mm': -12.0,
        'board_keepout_mm': 5.0,
        'travel': ((30.0, 170.0), (20.0, 160.0), (-12.0, 0.0)),
    })

    # 1) 规划阶段：模拟板位姿外轮廓右上角内缩，使右上角点 C038 落在 5mm 安全区外，规划直接换用安全区内的邻近点
    orig_fit = sess_mod.fit_board_to_module

    def skewed_fit(board_pts, module_pts, *, allow_scale=True):
        base_tr, info = orig_fit(board_pts, module_pts, allow_scale=allow_scale)

        def warped_tr(pts):
            out = np.asarray(base_tr(pts), dtype=float)
            single = out.ndim == 1
            arr = out.reshape(1, 2) if single else out.copy()
            # 仅使右上计划点 C038 (95, 10) 偏出 5mm 安全区（距外边约 4mm < 5mm），其余点与邻近点 C058 仍在安全区内
            for i, raw_p in enumerate(np.asarray(pts, dtype=float).reshape(-1, 2)):
                if abs(raw_p[0] - 95.0) < 1e-6 and abs(raw_p[1] - 10.0) < 1e-6:
                    arr[i] += [6.0, 6.0]
            return arr[0] if single else arr

        return warped_tr, info

    monkeypatch.setattr(sess_mod, 'fit_board_to_module', skewed_fit)
    s_plan = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path / 'plan', run=True)))
    s_plan.record.begin()
    start_pose = (95.0, 88.0, -3.0)
    s_plan.actual = start_pose
    s_plan.arrived(start_pose)
    issued_moves = []
    orig_move = s_plan.move
    monkeypatch.setattr(s_plan, 'move', lambda xyz: (issued_moves.append(tuple(xyz)), orig_move(xyz))[1])

    first_cmd = s_plan.advance([scene.render(start_pose, focus_z=-3.0)])
    assert first_cmd is not None
    assert 'C038' not in s_plan.plan
    assert all(s_plan._in_board_safe_zone(s_plan._target_camera_xy(lbl)) for lbl in s_plan.plan)
    assert all(s_plan._in_board_safe_zone(s_plan._target_camera_xy(lbl)) for lbl in s_plan.extra)
    assert len(s_plan.failed_points) == 1
    fp = s_plan.failed_points[0]
    assert fp['label'] == 'C038'
    assert fp['reason'] == '测点超出标定板安全区'
    assert fp['retest_label'] and fp['retest_label'] != 'C038'
    c038_cam = s_plan._target_camera_xy('C038')
    assert all(math.hypot(m[0] - c038_cam[0], m[1] - c038_cam[1]) > 1.0 for m in issued_moves)

    # 2) 运行中 locate & 居中迭代阶段：恢复正常拟合后，强行让某计划点/居中目标超出安全区，均不得下发该移动
    monkeypatch.undo()
    s_run = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path / 'run', run=True)))
    s_run.record.begin()
    s_run.actual = start_pose
    s_run.arrived(start_pose)
    run_moves = []
    orig_run_move = s_run.move
    monkeypatch.setattr(s_run, 'move', lambda xyz: (run_moves.append(tuple(xyz)), orig_run_move(xyz))[1])

    s_run.advance([scene.render(start_pose, focus_z=-3.0)])
    cur_lbl = s_run.label
    # 模拟下一个计划点突然因安全区变化落到安全区外 -> locate 不得下发该点移动，改切邻近安全点
    next_idx = next(i for i in range(s_run.point_index + 1, len(s_run.plan)) if s_run.plan[i] != s_run.ref_label)
    next_slot_lbl = s_run.plan[next_idx]
    bad_xy = s_run._target_camera_xy(next_slot_lbl)
    s_run.layout[next_slot_lbl] = s_run.board_outer_quad[0] + np.array([-2.0, 2.0])
    s_run.point_index = next_idx
    backup_move = s_run.locate(next_slot_lbl, -3.0)
    assert backup_move is not None
    assert s_run.label != next_slot_lbl
    assert s_run._in_board_safe_zone(backup_move[:2])
    assert any(f['label'] == next_slot_lbl and '测点超出标定板安全区' in f['reason'] for f in s_run.failed_points)
    assert all(math.hypot(m[0] - bad_xy[0], m[1] - bad_xy[1]) > 0.5 for m in run_moves if s_run.label != next_slot_lbl)

    # 3) 居中迭代时测得目标偏出安全区 -> 不下发该移动，该点失败并换补测点
    active_lbl = s_run.label
    s_run.actual = backup_move
    s_run.arrived(backup_move)
    monkeypatch.setattr(
        sess_mod, 'detect_charuco_corners',
        lambda img, det: (
            np.array([[2011.5, 1517.5], [100.0, 100.0], [200.0, 100.0], [300.0, 100.0], [100.0, 200.0], [200.0, 200.0]]),
            np.array([charuco_corner_id(active_lbl), 0, 1, 2, 3, 4]),
        ),
    )
    # 人为将 pixel_to_module 返回的角点坐标拉到板外（超过安全区），而画面中心仍在当前实际位姿附近
    orig_p2m = sess_mod.pixel_to_module
    monkeypatch.setattr(
        sess_mod, 'pixel_to_module',
        lambda u, v, pose, c: (np.array([500.0, 500.0, 0.0]) if abs(u - 2011.5) < 1e-6 and abs(v - 1517.5) < 1e-6
                               else orig_p2m(u, v, pose, c)),
    )
    # 触发居中迭代（令 tol 极小确保需要移动）
    s_run.cfg['center_tol_px'] = 0.0
    # 使 corner_uv 略偏离 center_uv
    monkeypatch.setattr(
        sess_mod, 'detect_charuco_corners',
        lambda img, det: (
            np.array([[2030.0, 1530.0], [100.0, 100.0], [200.0, 100.0], [300.0, 100.0], [100.0, 200.0], [200.0, 200.0]]),
            np.array([charuco_corner_id(active_lbl), 0, 1, 2, 3, 4]),
        ),
    )
    monkeypatch.setattr(
        sess_mod, 'pixel_to_module',
        lambda u, v, pose, c: (np.array([s_run.board_outer_quad[0][0] - 50.0, s_run.board_outer_quad[0][1] + 50.0, 0.0])
                               if abs(u - 2030.0) < 1e-6 else orig_p2m(u, v, pose, c)),
    )
    moves_before = len(run_moves)
    after_center_fail = s_run.advance([scene.render(backup_move, focus_z=-3.0)])
    assert after_center_fail is not None
    assert any(f['label'] == active_lbl and '测点超出标定板安全区' in f['reason'] for f in s_run.failed_points)
    # 下发的移动是下一个安全测点，而不是 500mm 外的出界移动
    assert s_run._in_board_safe_zone(run_moves[-1][:2])
    assert len(run_moves) == moves_before + 1


def test_charuco_centering_missing_id_or_too_few_corners_retests(tmp_path, monkeypatch):
    """居中时目标角点 ID 丢失或检出 ChArUco 角点数 < 6：不下发该点移动，记原因并走补测。"""
    import nest_clear.depthcal.session as sess_mod

    scene, cal, _, _ = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.015)
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, depth_calibration={
        **cfg.depth_calibration,
        'target': 'charuco',
        'camera_z_min_mm': -12.0,
        'travel': ((30.0, 170.0), (20.0, 160.0), (-12.0, 0.0)),
    })

    for mode in ('missing_id', 'too_few_corners'):
        sess = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path / mode, run=True)))
        sess.record.begin()
        start_pose = (118.0, 74.0, -3.0)
        sess.actual = start_pose
        sess.arrived(start_pose)
        first_cmd = sess.advance([scene.render(start_pose, focus_z=-3.0)])
        assert first_cmd is not None and sess.phase == 'center'
        failed_lbl = sess.label
        target_id = charuco_corner_id(failed_lbl)

        sess.actual = first_cmd
        sess.arrived(first_cmd)
        moves = []
        orig_move = sess.move
        monkeypatch.setattr(sess, 'move', lambda xyz: (moves.append(tuple(xyz)), orig_move(xyz))[1])

        if mode == 'missing_id':
            # 检出 >= 6 个角点，但缺少当前目标角点 ID
            fake_pts = np.array([[100.0 + 20.0 * i, 100.0] for i in range(8)], dtype=float)
            fake_ids = np.array([i for i in range(20, 28) if i != target_id][:8], dtype=int)
        else:
            # 包含目标角点 ID，但总角点数 < 6（仅 4 个）
            fake_pts = np.array([[2011.5, 1517.5], [100.0, 100.0], [200.0, 100.0], [300.0, 100.0]], dtype=float)
            fake_ids = np.array([target_id, 0, 1, 2], dtype=int)

        monkeypatch.setattr(sess_mod, 'detect_charuco_corners', lambda img, det, p=fake_pts, ids=fake_ids: (p, ids))
        backup_cmd = sess.advance([scene.render(first_cmd, focus_z=-3.0)])
        assert backup_cmd is not None
        assert len(sess.failed_points) == 1
        fp = sess.failed_points[0]
        assert fp['label'] == failed_lbl
        assert fp['error_code'] == 'DEPTH-TARGET'
        assert '画面中未见目标角点/标定板' in fp['reason']
        assert fp['retest_status'] == 'pending'
        assert sess.label == fp['retest_label'] and sess.label != failed_lbl
        assert len(moves) == 1
        assert moves[0][:2] == pytest.approx(sess._target_camera_xy(sess.label), abs=1e-3)
        monkeypatch.undo()


def test_charuco_e2e_tcp_160_to_161_normal_5_points(tmp_path):
    """正常 5 点 ChArUco 流程不受安全区保护影响：全部 5 点完成并回 161，无误报跳过。"""
    import time

    scene, cal, board_origin, board_rot = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.015)
    camera = SimCharucoDepthCamera(scene, board_origin, board_rot, start_xyz=(95.0, 88.0, -3.0))

    cfg = load_config('config/nest_clear.yaml')
    travel = ((32.0, 170.0), (20.0, 160.0), (-12.0, 0.0))
    settings = {
        **cfg.depth_calibration,
        'target': 'auto',
        'camera_z_min_mm': -9.0,
        'travel': travel,
        'coarse_step_mm': 1.0,
        'overshoot_mm': 0.2,
        'repeats_per_point': 2,
        'frames_per_step': 1,
        'fine_steps_target': 14,
        'settle_window_s': 0.0001,
        'arrival_timeout_s': 0.05,
    }
    cfg = replace(
        cfg,
        depth_calibration=settings,
        travel=travel,
        camera_calibration={**cfg.camera_calibration, 'board_thickness_mm': 3.15},
    )
    events = EventRecorder(tmp_path, run=True)

    with FakePLC() as plc:
        io = PLCClient('127.0.0.1', plc.port, 1, plc.contract)
        board = WorldBoard()
        clock = BTClock(board)
        plc_worker = PlcWorker(Handshake(io, plc.contract, events=events))
        for worker in (plc_worker, camera):
            clock.attach_worker(worker)
        batch = build_batch(board, cfg, [], RequestTracker(0), events, calibration=cal)
        clock.submit(batch)

        def exchange(xyz=None):
            if xyz is not None:
                camera.actual = tuple(xyz)
            plc.set_actual_xyz(camera.actual)
            task = plc.request(160, tool_id=65535)
            deadline = time.monotonic() + 8.0
            while batch.blackboard.read('committed_func') is None or events.request_pending or plc.io.read('result_id') != task:
                clock.tick_once()
                if time.monotonic() > deadline:
                    pytest.fail(f'未收到 Task_ID={task} 应答')
                time.sleep(0.0002)
            code = plc.io.read('reply_func')
            coords = plc.io.read_real32_block(40942, 3) if code == 160 else None
            return code, coords

        try:
            target = None
            session = None
            for _ in range(1200):
                code, target = exchange(target)
                if code == 161:
                    break
                assert code == 160, events.status.data.get('depthcal')
                session = batch.blackboard.read('depth_session')
                assert target is not None
                assert session._in_board_safe_zone(target[:2])
            else:
                pytest.fail('正常 5 点 ChArUco 160 会话未完成')
        finally:
            clock.shutdown()
            io.close()

    assert batch.blackboard.read('mode') == Mode.IDLE
    assert session is not None
    assert len(session.results) == 5
    assert session.failed_points == []
    assert set(session.plan) == {'C021', 'C038', 'C278', 'C261', 'C149'}
    assert session.board_fit['board_keepout_mm'] == 5.0


def test_charuco_centering_tolerates_0_3mm_outward_residual_while_locate_stays_strict(tmp_path, monkeypatch):
    """居中残差 0.3mm 偏向板外时不误判，过大出界仍拦截，locate 移到计划点仍用严格 0.05mm 边距。"""
    import nest_clear.depthcal.session as sess_mod
    from nest_clear.depthcal.layout import _inward_edge_normals

    scene, cal, _, _ = _shifted_scene(shift=(95.0, 88.0, -105.0), yaw_rad=0.015)
    cfg = load_config('config/nest_clear.yaml')
    cfg = replace(cfg, depth_calibration={
        **cfg.depth_calibration,
        'target': 'charuco',
        'camera_z_min_mm': -12.0,
        'board_keepout_mm': 5.0,
        'center_tol_px': 15.0,
        'travel': ((30.0, 170.0), (20.0, 160.0), (-12.0, 0.0)),
    })

    sess = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path / 'tol', run=True)))
    sess.record.begin()
    start_pose = (95.0, 88.0, -3.0)
    sess.actual = start_pose
    sess.arrived(start_pose)
    cmd = sess.advance([scene.render(start_pose, focus_z=-3.0)])
    assert cmd is not None and sess.phase == 'center'

    # 构造恰好比 5.0mm 严格安全区向板外偏 0.30mm（即距板外边 4.70mm）的模组 XY
    normals = _inward_edge_normals(sess.board_outer_quad)
    edge_pt = sess.board_outer_quad[0]
    edge_dir = sess.board_outer_quad[1] - sess.board_outer_quad[0]
    edge_dir = edge_dir / np.linalg.norm(edge_dir)
    mid_along_edge = edge_pt + edge_dir * 40.0
    # 沿内法线向内 4.70mm（比 keepout=5.0mm 向外偏 0.30mm），再沿边中线保证远离其余三边
    mod_03mm_out = mid_along_edge + normals[0] * 4.70
    cam_03mm_out = tuple((mod_03mm_out - sess.center_offset).tolist())

    # 严格边距判定（规划/locate）仍应判定出界
    assert not sess._in_board_safe_zone(cam_03mm_out)

    # 1) locate 移向计划测点仍使用严格边距：应拦截并不下发移动
    strict_lbl = next(lbl for lbl in sess.plan if lbl not in (sess.label, sess.ref_label))
    sess.layout[strict_lbl] = mod_03mm_out - sess.ref_xy + sess.layout[sess.ref_label]
    sess.point_index = sess.plan.index(strict_lbl)
    sess.locate(strict_lbl, -3.0)
    assert any(f['label'] == strict_lbl and '测点超出标定板安全区' in f['reason'] for f in sess.failed_points)

    # 2) 恢复正常会话测试居中迭代：残差 0.30mm 偏向板外时，居中阶段放宽容差 max(0.05, center_tol_mm * 1.5) ≈ 0.53mm，不得误判
    sess2 = Session(cfg, cal, DepthRecord(EventRecorder(tmp_path / 'center_ok', run=True)))
    sess2.record.begin()
    sess2.actual = start_pose
    sess2.arrived(start_pose)
    cmd2 = sess2.advance([scene.render(start_pose, focus_z=-3.0)])
    assert cmd2 is not None and sess2.phase == 'center'
    active_lbl = sess2.label
    active_id = charuco_corner_id(active_lbl)
    sess2.actual = cmd2
    sess2.arrived(cmd2)

    # 让角点像素偏离中心 20 px（> center_tol_px=15，触发居中移动），且实测角点模组坐标落在距板边 4.70mm（偏向板外 0.30mm）
    normals2 = _inward_edge_normals(sess2.board_outer_quad)
    e0, e1 = sess2.board_outer_quad[0], sess2.board_outer_quad[1]
    u_dir = (e1 - e0) / np.linalg.norm(e1 - e0)
    mod_center_03mm_out = e0 + u_dir * 40.0 + normals2[0] * 4.70
    monkeypatch.setattr(
        sess_mod, 'detect_charuco_corners',
        lambda img, det: (
            np.array([[2031.5, 1517.5], [100.0, 100.0], [200.0, 100.0], [300.0, 100.0], [100.0, 200.0], [200.0, 200.0]]),
            np.array([active_id, 0, 1, 2, 3, 4]),
        ),
    )
    orig_p2m = sess_mod.pixel_to_module
    monkeypatch.setattr(
        sess_mod, 'pixel_to_module',
        lambda u, v, pose, c: (
            np.array([mod_center_03mm_out[0], mod_center_03mm_out[1], 0.0])
            if abs(u - 2031.5) < 1e-6 and abs(v - 1517.5) < 1e-6 else orig_p2m(u, v, pose, c)
        ),
    )
    move_ok = sess2.advance([scene.render(cmd2, focus_z=-3.0)])
    assert sess2.phase == 'center' and sess2.center_round == 2
    assert sess2.failed_points == []
    assert move_ok[:2] == pytest.approx((mod_center_03mm_out - sess2.center_offset).tolist(), abs=1e-4)

    # 3) 若居中目标偏出严格边距 1.00mm（超出放宽容差 ~0.53mm），仍应触发 DEPTH-SAFE-ZONE 拦截
    mod_center_1mm_out = e0 + u_dir * 40.0 + normals2[0] * 4.00
    monkeypatch.setattr(
        sess_mod, 'pixel_to_module',
        lambda u, v, pose, c: (
            np.array([mod_center_1mm_out[0], mod_center_1mm_out[1], 0.0])
            if abs(u - 2031.5) < 1e-6 and abs(v - 1517.5) < 1e-6 else orig_p2m(u, v, pose, c)
        ),
    )
    sess2.actual = move_ok
    sess2.arrived(move_ok)
    sess2.advance([scene.render(move_ok, focus_z=-3.0)])
    assert any(f['label'] == active_lbl and f['error_code'] == 'DEPTH-SAFE-ZONE' for f in sess2.failed_points)

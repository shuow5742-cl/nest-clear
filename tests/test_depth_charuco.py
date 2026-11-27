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
from nest_clear.depthcal.layout import charuco_corner_id
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
    """depth_calibration.target 默认为 auto，支持 auto/film/charuco 并拒绝非法值。"""
    cfg = load_config('config/nest_clear.yaml')
    assert cfg.depth_calibration['target'] == 'auto'

    raw = yaml.safe_load(Path('config/nest_clear.yaml').read_text(encoding='utf-8'))
    for val in ('auto', 'film', 'charuco'):
        good = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'target': val}}
        p = tmp_path / f'{val}.yaml'
        p.write_text(yaml.safe_dump(good), encoding='utf-8')
        assert load_config(p, require_tools=False).depth_calibration['target'] == val

    bad = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'target': 'chessboard'}}
    p_bad = tmp_path / 'bad.yaml'
    p_bad.write_text(yaml.safe_dump(bad), encoding='utf-8')
    with pytest.raises(GuardError, match='depth_calibration.target'):
        load_config(p_bad, require_tools=False)


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

"""160 请求的安全配置、保留码、模式互斥与预览释放。"""

from dataclasses import replace
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nest_clear.config import load_config
from nest_clear.depthcal.record import DepthRecord
from nest_clear.depthcal.session import Session
from nest_clear.flow.modes import Mode
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import guard_position
from nest_clear.preview.control import REQUEST_CAMERAS
from nest_clear.preview.eligibility import preview_eligibility
from nest_clear.record.events import EventRecorder
from nest_clear.vision.geometry import Calibration
from test_tooloffset_flow import rig  # noqa: F401


def test_default_config_z_min_is_minus_12():
    cfg = load_config('config/nest_clear.yaml')
    assert cfg.depth_calibration['camera_z_min_mm'] == -12.0
    assert cfg.depth_calibration['travel'] == ((32.0, 158.6), (-15.0, 201.0), (-12.0, 0.0))
    assert cfg.travel == ((80.0, 180.0), (-10.0, 150.0), (-15.0, 0.0))
    assert cfg.depth_calibration['fine_range_mode'] == 'auto'
    assert cfg.depth_calibration['fine_range_factor'] == 0.75
    assert cfg.depth_calibration['fine_range_min_mm'] == 0.3
    assert cfg.depth_calibration['fine_range_max_mm'] == 3.0
    assert cfg.depth_calibration['fine_steps_target'] == 40
    assert cfg.depth_calibration['fine_step_min_mm'] == 0.02
    assert cfg.depth_calibration['fine_step_max_mm'] == 0.1
    assert cfg.depth_calibration['dof_ratio'] == 0.8


def test_depth_protocol_missing_z_min_and_reserved_codes(rig):
    rig.cfg.depth_calibration['camera_z_min_mm'] = None
    assert rig.request(1, 160, None) == (999, {})
    assert '未配置相机 Z 下限' in rig.plc.diagnostic['what']
    for task, code in enumerate((170, 171, 180, 181, 190, 191), 2):
        assert rig.request(task, code, None) == (999, {})
        assert rig.plc.diagnostic['what'] == '下针深度标定 170/180/190 尚未实现'
    rig.batch.blackboard.write('mode', Mode.PICKING, 'test')
    assert rig.request(8, 160, None) == (999, {})
    assert rig.batch.blackboard.read('mode') == Mode.PICKING
    rig.batch.blackboard.write('mode', Mode.DEPTH_CALIB, 'test')
    assert rig.request(9, 100, 65535) == (999, {})
    assert rig.batch.blackboard.read('mode') == Mode.IDLE


def test_depth_preview_ownership_consistent_across_hosts():
    assert REQUEST_CAMERAS[160] == frozenset({'main'})
    sup = {'pid': 1, 'child_pid': 2, 'child_state': '就绪', 'word': 0}
    allowed_main, reason = preview_eligibility(sup, 0, Mode.DEPTH_CALIB.value, 'main')
    assert not allowed_main and '下针深度标定中' in reason
    assert preview_eligibility(sup, 0, Mode.DEPTH_CALIB.value, 'front') == (True, None)
    assert preview_eligibility(sup, 0, Mode.DEPTH_CALIB.value, 'side') == (True, None)
    # 停机/手动时允许预览
    assert preview_eligibility({**sup, 'word': 1}, 0, Mode.DEPTH_CALIB.value, 'main') == (True, None)


def test_160_releases_main_preview(rig):
    closed = []
    preview_on = [True]
    def stop():
        if preview_on[0]:
            preview_on[0] = False
            closed.append('main')
            return True
        return False
    rig.events.stop_main_preview = stop
    rig.cfg.depth_calibration['camera_z_min_mm'] = -12.0
    code, coords = rig.request(1, 160, None)
    assert code == 160 and 'scan_xyz' in coords
    assert closed == ['main']


def test_depth_travel_config_validation_and_fallback(tmp_path, caplog):
    raw = yaml.safe_load(Path('config/nest_clear.yaml').read_text(encoding='utf-8'))

    # 缺省时回退全局 travel 并在配置加载日志给出说明
    no_travel = {**raw, 'depth_calibration': {k: v for k, v in raw['depth_calibration'].items() if k != 'travel'}}
    p_fallback = tmp_path / 'fallback.yaml'
    p_fallback.write_text(yaml.safe_dump(no_travel), encoding='utf-8')
    with caplog.at_level(logging.INFO, logger='nest_clear.config'):
        cfg_fb = load_config(p_fallback, require_tools=False)
    assert cfg_fb.depth_calibration['travel'] is None
    assert 'depth_calibration.travel 未配置，下针深度标定将回退使用全局 travel' in caplog.text

    import numpy as np
    cal = Calibration(np.eye(3), np.zeros(5), np.eye(3), np.zeros(3), -100.0)
    session_fb = Session(cfg_fb, cal, DepthRecord(EventRecorder(tmp_path, run=False)))
    assert session_fb.travel == cfg_fb.travel
    assert session_fb.z_min == max(cfg_fb.travel[2][0], cfg_fb.depth_calibration['camera_z_min_mm'])

    # 配置独立行程后，z_min 仍取 max(travel z 下限, camera_z_min_mm)
    custom = {
        **raw,
        'depth_calibration': {
            **raw['depth_calibration'],
            'travel': {'x': [32.0, 158.6], 'y': [-15.0, 201.0], 'z': [-14.0, 0.0]},
            'camera_z_min_mm': -12.0,
        },
    }
    p_custom = tmp_path / 'custom.yaml'
    p_custom.write_text(yaml.safe_dump(custom), encoding='utf-8')
    cfg_custom = load_config(p_custom, require_tools=False)
    session_custom = Session(cfg_custom, cal, DepthRecord(EventRecorder(tmp_path, run=False)))
    assert session_custom.travel == ((32.0, 158.6), (-15.0, 201.0), (-14.0, 0.0))
    assert session_custom.z_min == -12.0

    # 非法范围（min >= max、NaN/Inf、缺少轴）均在配置加载时报错
    for bad_travel in (
        {'x': [100.0, 100.0], 'y': [-15.0, 201.0], 'z': [-12.0, 0.0]},
        {'x': [160.0, 32.0], 'y': [-15.0, 201.0], 'z': [-12.0, 0.0]},
        {'x': [32.0, float('inf')], 'y': [-15.0, 201.0], 'z': [-12.0, 0.0]},
        {'x': [32.0, float('nan')], 'y': [-15.0, 201.0], 'z': [-12.0, 0.0]},
        {'x': [32.0, 158.6], 'y': [-15.0, 201.0]},
    ):
        bad = {**raw, 'depth_calibration': {**raw['depth_calibration'], 'travel': bad_travel}}
        p_bad = tmp_path / 'bad.yaml'
        p_bad.write_text(yaml.safe_dump(bad), encoding='utf-8')
        with pytest.raises(GuardError, match='depth_calibration.travel'):
            load_config(p_bad, require_tools=False)


def test_other_flows_still_use_global_travel(rig):
    """全局行程外、深度行程内的坐标（X=40, Y=180）在其他流程仍受全局 travel 限制。"""
    from nest_clear.flow.picking import calculate_pick_coords
    from nest_clear.tooloffset.alignment import ToolAlignmentSession

    # 1) 挑取/扫描通用守卫与挑取坐标计算仍使用全局 travel
    with pytest.raises(GuardError, match='POSITION-TRAVEL'):
        guard_position((40.0, 180.0, -5.0), rig.cfg.travel, '挑取/扫描')
    with pytest.raises(GuardError, match='POSITION-TRAVEL'):
        calculate_pick_coords((40.0, 180.0, -5.0), rig.cfg.tools, rig.cfg.pick_depth_z, rig.cfg.travel, task_id=1)

    # 2) 工具偏移对准仍使用全局 travel，对 (40, 180, -5) 下探步报错超出行程
    align = ToolAlignmentSession(1, (40.0, 180.0), rig.cfg.travel, rig.cfg.tool_offset['align'])
    out = align.step(
        (40.0, 180.0, -5.0),
        {'front': (320.0, 200.0), 'side': (320.0, 200.0)},
        {'front': (320.0, 400.0), 'side': (320.0, 400.0)},
    )
    assert out['status'] == 'failed' and '超出行程' in out['reason']

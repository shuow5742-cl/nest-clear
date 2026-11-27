"""160 请求的安全配置、保留码、模式互斥与预览释放。"""

from dataclasses import replace

from nest_clear.config import load_config
from nest_clear.flow.modes import Mode
from nest_clear.preview.control import REQUEST_CAMERAS
from nest_clear.preview.eligibility import preview_eligibility
from test_tooloffset_flow import rig  # noqa: F401


def test_default_config_z_min_is_minus_12():
    cfg = load_config('config/nest_clear.yaml')
    assert cfg.depth_calibration['camera_z_min_mm'] == -12.0
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

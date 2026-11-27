"""160 请求的默认安全行为与保留码。"""

from nest_clear.flow.modes import Mode
from test_tooloffset_flow import rig  # noqa: F401


def test_depth_protocol(rig):
    assert rig.cfg.depth_calibration['camera_z_min_mm'] is None
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

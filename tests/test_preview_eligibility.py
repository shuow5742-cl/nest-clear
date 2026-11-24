import pytest

from nest_clear.flow.modes import Mode
from nest_clear.preview.eligibility import CAMERAS, MODE_CAMERAS, preview_eligibility


@pytest.mark.parametrize(('mode', 'locked'), [
    ('空闲', set()), ('挑毛中', {'main'}), ('相机标定中', {'main'}),
    ('对刀基准建立中', {'front', 'side'}),
])
def test_expected_camera_ownership(mode, locked):
    supervisor = {'pid': 1, 'child_pid': 2, 'word': 0}
    assert {camera for camera in CAMERAS if not preview_eligibility(supervisor, 0, mode, camera)[0]} == locked


@pytest.mark.parametrize('camera', CAMERAS)
@pytest.mark.parametrize('bit', [0, 4, 6])
def test_device_overrides_picking_mode(bit, camera):
    assert preview_eligibility({'pid': 1, 'child_pid': 2, 'word': 1 << bit}, 1, '挑毛中', camera) == (True, None)


@pytest.mark.parametrize('camera', CAMERAS)
@pytest.mark.parametrize('mode', list(Mode))
def test_all_modes(mode, camera):
    allowed, reason = preview_eligibility({'pid': 1, 'child_pid': 2, 'word': 0}, 1, mode.value, camera)
    assert allowed == (camera not in MODE_CAMERAS[mode.value])
    assert (reason is None) == allowed
    if reason:
        assert mode.value in reason


@pytest.mark.parametrize('camera', CAMERAS)
def test_missing_stale_parent_and_stopped_child(camera):
    assert preview_eligibility({'pid': 1}, 3.01, '空闲', camera) == (False, '父程序未运行')
    assert preview_eligibility({}, None, '空闲', camera) == (False, '父程序未运行')
    assert preview_eligibility({'pid': 1, 'word': 0}, 3, '挑毛中', camera) == (True, None)

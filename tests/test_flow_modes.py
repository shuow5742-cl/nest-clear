"""验证工作模式的全部有序切换组合。"""

import pytest

from nest_clear.flow.modes import Mode, transition
from nest_clear.guard import GuardError


@pytest.mark.parametrize("current", list(Mode))
@pytest.mark.parametrize("target", list(Mode))
def test_transitions(current: Mode, target: Mode) -> None:
    """空闲可进入业务模式，业务模式只能结束回空闲。"""
    allowed = (current is Mode.IDLE and target is not Mode.IDLE) or (
        current is not Mode.IDLE and target is Mode.IDLE
    )
    if allowed:
        assert transition(current, target) is target
    else:
        with pytest.raises(GuardError, match="MODE-TRANSITION"):
            transition(current, target)

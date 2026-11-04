"""主程序工作模式及允许的切换规则，不依赖设备或行为树。"""

from enum import Enum

from nest_clear.guard import GuardError


class Mode(Enum):
    """区分空闲、挑毛和两把工具各自的对刀会话。"""

    IDLE = "空闲"
    PICKING = "挑毛中"
    CALIBRATING_1 = "对刀中(工具1)"
    CALIBRATING_2 = "对刀中(工具2)"


_ALLOWED = {
    Mode.IDLE: frozenset((Mode.PICKING, Mode.CALIBRATING_1, Mode.CALIBRATING_2)),
    Mode.PICKING: frozenset((Mode.IDLE,)),
    Mode.CALIBRATING_1: frozenset((Mode.IDLE,)),
    Mode.CALIBRATING_2: frozenset((Mode.IDLE,)),
}


def transition(current: Mode, target: Mode) -> Mode:
    """检查一次实际模式切换；非法切换交由调用方记录和处理。"""
    if not isinstance(current, Mode) or not isinstance(target, Mode) or target not in _ALLOWED[current]:
        raise GuardError(
            "MODE-TRANSITION",
            f"不允许从 {current} 切换到 {target}",
            "工作模式",
            "先结束当前挑毛或对刀会话，再开始新会话。",
        )
    return target

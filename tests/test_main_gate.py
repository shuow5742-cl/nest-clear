"""验证主循环暂停期间仍保活，恢复后推进树。"""
from nest_clear.main import Application


class Gate:
    """可控暂停门，记录打点次数。"""
    def __init__(self) -> None:
        """初始化暂停和打点状态。"""
        self.count = 0
        self.value = True

    def beat(self) -> None:
        """累计一次活性打点。"""
        self.count += 1

    def paused(self) -> bool:
        """返回测试指定的暂停状态。"""
        return self.value


def test_gate_keeps_beating_without_ticks() -> None:
    """暂停不 tick，解除暂停后立即恢复调度。"""
    class Clock:
        """只记录时钟推进次数。"""
        count = 0

        def tick_once(self) -> None:
            """累计推进次数。"""
            self.count += 1

    class Handle:
        """表示正在运行的批次。"""
        result = None

    app = Application.__new__(Application)
    app.clock = Clock()
    app.handle = Handle()
    gate = Gate()
    app.step(gate)
    app.step(gate)
    assert gate.count == 2
    assert app.clock.count == 0
    gate.value = False
    app.step(gate)
    assert gate.count == 3
    assert app.clock.count == 1

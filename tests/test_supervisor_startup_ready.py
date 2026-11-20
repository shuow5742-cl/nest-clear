"""验证父程序启动时清理 PLC 遗留的就绪标志。"""

from dataclasses import replace

from nest_clear.config import load_config
from nest_clear.supervisor.runner import Supervisor


class MemoryPLC:
    """模拟断线和寄存器写入，不建立网络连接。"""

    def __init__(self) -> None:
        """初始化连接状态和写入记录。"""
        self.connected = True
        self.writes = []

    def read(self, name: str) -> int:
        """返回空闲状态，断线时模拟读取失败。"""
        if not self.connected:
            raise OSError("断线")
        return 0

    def write(self, name: str, value: int) -> None:
        """记录成功写入，断线时模拟写入失败。"""
        if not self.connected:
            raise OSError("断线")
        self.writes.append((name, value))

    def close(self) -> None:
        """模拟关闭连接。"""


def test_run_clears_ready_before_first_tick(tmp_path, monkeypatch) -> None:
    """启动清零发生在首拍之前，并记录原因。"""
    config = replace(load_config("config/nest_clear.yaml"), record_directory=str(tmp_path))
    plc = MemoryPLC()
    supervisor = Supervisor(config, plc)
    events = []
    monkeypatch.setattr(supervisor.events, "emit", lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(supervisor, "_ui_tick", lambda: None)

    def stop_at_tick() -> None:
        """检查首拍前的写入，然后结束循环。"""
        assert plc.writes == [("algorithm_ready", 0)]
        raise KeyboardInterrupt

    monkeypatch.setattr(supervisor, "tick", stop_at_tick)
    monkeypatch.setattr(supervisor, "_reply_after_failure", lambda *args, **kwargs: None)
    supervisor.run()
    assert ("algorithm_ready_cleared", {"reason": "supervisor_start"}) in events


def test_startup_retry_waits_for_connection_and_no_child(tmp_path, monkeypatch) -> None:
    """断线后在无子进程时补写一次，成功后不再重复。"""
    config = replace(load_config("config/nest_clear.yaml"), record_directory=str(tmp_path))
    plc = MemoryPLC()
    plc.connected = False
    supervisor = Supervisor(config, plc)
    events = []
    monkeypatch.setattr(supervisor.events, "emit", lambda event, **fields: events.append((event, fields)))
    supervisor._startup_ready_pending = True
    supervisor._clear_startup_ready()
    supervisor.tick()
    plc.connected = True

    class Child:
        """模拟仍在运行的主程序。"""

        def poll(self) -> None:
            """报告进程仍在运行。"""
            return None

    supervisor.child = Child()
    monkeypatch.setattr(supervisor, "_publish", lambda: True)
    supervisor.tick()
    assert ("algorithm_ready", 0) not in plc.writes
    supervisor.child = None
    supervisor.tick()
    supervisor.tick()
    assert plc.writes.count(("algorithm_ready", 0)) == 1
    assert events.count(("algorithm_ready_cleared", {"reason": "supervisor_start"})) == 1

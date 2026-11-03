"""验证主程序暂停门禁在异常文件和回调下仍可运行。"""
import json

from nest_clear.supervisor.gate import SupervisorGate


def test_gate(tmp_path) -> None:
    """缺文件不暂停，暂停状态变化触发标记并继续打活性点。"""
    events = []
    gate = SupervisorGate(tmp_path, mark=lambda name, **kw: events.append(name))
    assert not gate.paused()
    gate.beat()
    assert (tmp_path / "main.alive").exists()
    state = tmp_path / "supervisor.json"
    state.write_text(json.dumps({"paused": True}))
    assert gate.paused()
    state.write_text("无效")
    assert gate.paused()
    state.write_text(json.dumps({"paused": False}))
    assert not gate.paused()
    assert events == ["device_pause_begin", "device_pause_end"]
    state.unlink()
    assert not gate.paused()


def test_gate_never_raises(tmp_path) -> None:
    """目录不可写与标记回调异常都不向主程序传播。"""
    path = tmp_path / "file"
    path.write_text("x")
    gate = SupervisorGate(path)
    gate.beat()
    assert not gate.paused()
    def bad_mark(*args, **kwargs):
        """模拟记录层故障。"""
        raise RuntimeError("错误")
    gate = SupervisorGate(tmp_path, mark=bad_mark)
    (tmp_path / "supervisor.json").write_text('{"paused": true}')
    assert gate.paused()

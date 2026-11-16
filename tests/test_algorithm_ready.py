"""不依赖网络验证父程序的算法就绪标志时序。"""

from dataclasses import replace
from unittest.mock import Mock

from nest_clear.config import load_config
from nest_clear.supervisor.runner import Supervisor


def test_supervisor_clears_ready_before_start(tmp_path, monkeypatch) -> None:
    """启动子进程之前必须先清除 PLC 上遗留的就绪标志。"""
    calls = []
    plc = Mock()
    plc.write.side_effect = lambda name, value: calls.append((name, value))
    child = Mock(pid=42)
    monkeypatch.setattr("nest_clear.supervisor.runner.subprocess.Popen", lambda command: calls.append("start") or child)
    config = replace(load_config("config/nest_clear.yaml"), record_directory=str(tmp_path))
    supervisor = Supervisor(config, plc)
    supervisor._start()
    assert calls == [("algorithm_ready", 0), "start"]


def test_supervisor_clears_ready_after_stop_and_exit(tmp_path) -> None:
    """主动停止和意外退出均清除就绪标志。"""
    plc = Mock()
    config = replace(load_config("config/nest_clear.yaml"), record_directory=str(tmp_path))
    supervisor = Supervisor(config, plc)
    child = Mock(pid=42)
    child.poll.return_value = None
    supervisor.child = child
    assert supervisor._stop("测试")
    child.wait.assert_called_once()
    plc.write.assert_called_with("algorithm_ready", 0)
    plc.reset_mock()
    child.poll.return_value = 1
    supervisor.child = child
    plc.read.return_value = 0
    assert not supervisor._running()
    plc.write.assert_called_once_with("algorithm_ready", 0)

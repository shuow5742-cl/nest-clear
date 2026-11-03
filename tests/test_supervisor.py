"""验证状态机、真实 PLC 通信及主进程生命周期。"""
from dataclasses import replace
import json
import sys
import time

from nest_clear.comm import PLCClient, load_contract
from nest_clear.config import load_config
from nest_clear.supervisor.device_state import DeviceStateTracker, describe
from nest_clear.supervisor.runner import Supervisor
from sim.fake_plc import FakePLC


def test_state_machine() -> None:
    """验证边沿、急停优先以及暂停只能由启动位解除。"""
    tracker = DeviceStateTracker()
    assert tracker.decide(1 << 7, child_running=False).action == "start"
    assert tracker.decide(1 << 7, child_running=True).action is None
    assert tracker.decide(0, child_running=True).action is None
    assert tracker.decide(1 << 7, child_running=True).action == "restart"
    assert tracker.decide((1 << 6) | (1 << 7), child_running=True).action == "estop"
    assert tracker.decide(1 << 7, child_running=False).action is None
    assert tracker.decide(1 << 3, child_running=True).paused
    assert tracker.decide(0, child_running=True).paused
    assert tracker.decide((1 << 4) | (1 << 2), child_running=True).paused
    assert not tracker.decide(1 << 2, child_running=True).paused
    assert describe((1 << 9) | (1 << 10) | (1 << 11)) == ["信息提示", "一般报警", "严重故障"]
    tracker.decide(0, child_running=False)
    assert tracker.decide((1 << 6) | (1 << 7), child_running=False).action is None
    assert tracker.decide(1 << 7, child_running=False).action is None


def test_plc_and_child(tmp_path, caplog) -> None:
    """使用真实 TCP 模拟 PLC 和假进程检验启动、暂停、心跳及急停。"""
    script = tmp_path / "child.py"
    script.write_text("import time\nwhile True:\n    time.sleep(0.05)\n")
    config = load_config("config/nest_clear.yaml")
    config = replace(config, record_directory=str(tmp_path), supervisor=replace(config.supervisor, child_command=(sys.executable, str(script))))
    with FakePLC() as fake:
        plc = PLCClient(fake.host, fake.port, 1, load_contract("config/plc_contract.yaml"))
        supervisor = Supervisor(config, plc)
        try:
            fake.set_status(1 << 7)
            supervisor.tick()
            assert supervisor.child is not None and supervisor.child.poll() is None
            first = fake.io.read("heartbeat")
            supervisor._next_heartbeat = 0
            supervisor.tick()
            assert fake.io.read("heartbeat") == first + 1
            fake.set_status(1 << 3)
            supervisor.tick()
            assert json.loads((tmp_path / "supervisor.json").read_text())["paused"]
            fake.set_status(1 << 6)
            supervisor.tick()
            assert supervisor.child is None
            fake.set_status(0)
            supervisor.tick()
            assert supervisor.child is None
            fake.set_status((1 << 7) | (1 << 11))
            supervisor.tick()
            assert supervisor.child is not None
            assert "严重故障" in caplog.text
        finally:
            supervisor._stop("测试结束")
            plc.close()


def test_watchdog_does_not_restart(tmp_path) -> None:
    """活性超时后停止进程，直到新初始化边沿才再次启动。"""
    class PLC:
        """提供可控状态字，隔离看门狗处置。"""
        word = 128
        def read(self, name: str) -> int:
            """返回当前状态字。"""
            return self.word
        def write(self, name: str, value: int) -> None:
            """接受心跳。"""
    cfg = load_config("config/nest_clear.yaml")
    cfg = replace(cfg, record_directory=str(tmp_path), supervisor=replace(cfg.supervisor, watchdog_s=0.01, child_command=(sys.executable, "-c", "import time; time.sleep(30)")))
    plc = PLC()
    sup = Supervisor(cfg, plc)
    try:
        sup.tick()
        (tmp_path / "main.alive").write_text("alive")
        time.sleep(0.03)
        plc.word = 0
        sup.tick()
        assert sup.child is None
        sup.tick()
        assert sup.child is None
    finally:
        sup._stop("测试结束")


def test_read_failure_keeps_state_and_recovers(tmp_path, caplog) -> None:
    """断线不伪造状态或动作，恢复后继续按最后成功状态识别边沿。"""
    class PLC:
        """模拟暂时断线的具名接口。"""
        word = 8
        failed = False
        def read(self, name: str) -> int:
            """按开关模拟读失败。"""
            if self.failed:
                raise OSError("断线")
            return self.word
        def write(self, name: str, value: int) -> None:
            """接受心跳。"""
    cfg = replace(load_config("config/nest_clear.yaml"), record_directory=str(tmp_path))
    plc = PLC()
    sup = Supervisor(cfg, plc)
    sup.tick()
    plc.failed = True
    sup.tick()
    assert sup.word == 8 and sup.paused
    assert "查网线/IP/PLC 是否上电" in caplog.text
    plc.failed = False
    plc.word = 4
    sup.tick()
    assert not sup.paused


def test_startup_grace_kills_and_replies(tmp_path, caplog) -> None:
    """首次打点迟迟不到时停止主程序，并向等待中的 PLC 回错误。"""
    config = load_config("config/nest_clear.yaml")
    config = replace(
        config,
        record_directory=str(tmp_path),
        supervisor=replace(
            config.supervisor,
            startup_grace_s=0.01,
            child_command=(sys.executable, "-c", "import time; time.sleep(30)"),
        ),
    )
    with FakePLC() as fake:
        plc = PLCClient(fake.host, fake.port, 1, fake.contract)
        sup = Supervisor(config, plc)
        try:
            fake.set_status(128)
            fake.io.write("task_id", 17)
            sup.tick()
            time.sleep(0.03)
            sup.tick()
            assert sup.child is None
            assert fake.io.read("reply_func") == 999
            assert fake.io.read("result_id") == 17
            assert "仍未打出活性点" in caplog.text
            assert "查 main.log" in caplog.text
        finally:
            sup._stop("测试结束")
            plc.close()


def test_unexpected_exit_replies_in_commit_order(tmp_path, caplog) -> None:
    """意外死亡时通过真实握手先写 999，最后提交任务编号。"""
    cfg = load_config("config/nest_clear.yaml")
    cfg = replace(
        cfg,
        record_directory=str(tmp_path),
        supervisor=replace(cfg.supervisor, child_command=(sys.executable, "-c", "import sys; sys.exit(7)")),
    )
    with FakePLC() as fake:
        plc = PLCClient(fake.host, fake.port, 1, fake.contract)
        sup = Supervisor(cfg, plc)
        try:
            fake.set_status(128)
            fake.io.write("task_id", 23)
            sup.tick()
            assert sup.child is not None
            sup.child.wait(timeout=3)
            fake.writes.clear()
            sup.tick()
            assert fake.io.read("reply_func") == 999
            assert fake.io.read("result_id") == 23
            relevant = [(address, data) for address, data in fake.writes if address in (
                fake.contract["reply_func"].address, fake.contract["result_id"].address,
            )]
            assert relevant == [
                (fake.contract["reply_func"].address, [999]),
                (fake.contract["result_id"].address, [23]),
            ]
            assert "退出码 7" in caplog.text and "Task_ID=23" in caplog.text
        finally:
            sup._stop("测试结束")
            plc.close()


def test_intentional_stop_does_not_reply(tmp_path) -> None:
    """急停、初始化重启和父程序停止都不代替主程序回复。"""
    cfg = load_config("config/nest_clear.yaml")
    cfg = replace(
        cfg, record_directory=str(tmp_path),
        supervisor=replace(cfg.supervisor, child_command=(sys.executable, "-c", "import time; time.sleep(30)")),
    )
    with FakePLC() as fake:
        plc = PLCClient(fake.host, fake.port, 1, fake.contract)
        sup = Supervisor(cfg, plc)
        try:
            fake.io.write("task_id", 24)
            fake.set_status(128)
            sup.tick()
            fake.set_status(0)
            sup.tick()
            fake.set_status(128)
            sup.tick()
            fake.set_status(64)
            sup.tick()
            assert fake.io.read("result_id") == 0
            assert fake.io.read("reply_func") == 0
            fake.set_status(0)
            sup.tick()
            fake.set_status(128)
            sup.tick()
            sup._stop("父程序退出")
            assert fake.io.read("result_id") == 0
            assert fake.io.read("reply_func") == 0
        finally:
            sup._stop("测试结束")
            plc.close()


def test_unexpected_exit_without_pending_request(tmp_path) -> None:
    """请求已答复或 Task_ID 为零时不能重复提交 999。"""
    cfg = load_config("config/nest_clear.yaml")
    cfg = replace(cfg, record_directory=str(tmp_path))
    with FakePLC() as fake:
        plc = PLCClient(fake.host, fake.port, 1, fake.contract)
        sup = Supervisor(cfg, plc)
        try:
            for task_id, result_id in ((31, 31), (0, 31)):
                fake.io.write("task_id", task_id)
                fake.io.write("result_id", result_id)
                fake.writes.clear()
                sup._reply_after_failure("意外退出")
                assert fake.io.read("result_id") == result_id
                assert fake.io.read("reply_func") == 0
                assert not fake.writes
        finally:
            plc.close()

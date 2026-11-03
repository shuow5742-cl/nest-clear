"""验证组合树的通信恢复、结构日志与功能码扩展。"""
from dataclasses import replace
import logging
from pathlib import Path
import time

import pytest

from nest_clear.config import load_config
from nest_clear.flow.batch import BRANCHES, format_tree
from nest_clear.flow.leaves import PrepareScan
from nest_clear.guard import GuardError
from nest_clear.main import Application
from sim.fake_plc import FakePLC

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """创建尚未 tick 的主程序，方便测试先注入通信故障。"""
    caplog.set_level(logging.INFO, logger="nest_clear.main")
    with FakePLC() as plc:
        config = load_config(ROOT / "config/nest_clear.yaml")
        config = replace(
            config,
            plc=replace(config.plc, ip=plc.host, port=plc.port),
            vision=replace(config.vision, camera_backend="mock"),
            record_directory=str(tmp_path),
        )
        app = Application(config, ROOT / "config/nest_clear.yaml")
        try:
            yield plc, app
        finally:
            app.close()


def drive_until(app: Application, condition: object, timeout: float = 5) -> None:
    """在当前线程推进时钟，直到测试条件成立。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.step()
        if condition():
            return
        time.sleep(.002)
    pytest.fail("等待行为树条件超时")


def test_poll_failure_recovers(prepared: tuple, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """首次轮询失败只告警；下次重连成功后仍处理同一请求。"""
    plc, app = prepared
    original = app.plc.handshake.read_request
    attempts = []

    def read() -> tuple[int, int]:
        """第一次抛出超时，其后使用真实 TCP 读取。"""
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("临时断线")
        return original()

    monkeypatch.setattr(app.plc.handshake, "read_request", read)
    task = plc.request(1)
    with caplog.at_level(logging.WARNING):
        drive_until(app, lambda: plc.io.read("result_id") == task)
    assert len(attempts) >= 2
    assert plc.io.read("reply_func") == 1
    assert "临时断线" in caplog.text


def test_reply_retries_three_attempts(prepared: tuple, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """两次写失败后第三次提交；日志中的失败带任务编号。"""
    plc, app = prepared
    original = app.plc.handshake.write_reply
    attempts = []

    def write(task: int, func: int, coordinates: object) -> None:
        """前两次模拟网络中断，第三次真实写入。"""
        attempts.append(task)
        if len(attempts) < 3:
            raise TimeoutError("写入超时")
        original(task, func, coordinates)

    monkeypatch.setattr(app.plc.handshake, "write_reply", write)
    task = plc.request(1)
    with caplog.at_level(logging.WARNING):
        drive_until(app, lambda: plc.io.read("result_id") == task)
    assert attempts == [task, task, task]
    assert f"Task_ID={task}" in caplog.text
    assert plc.io.read("reply_func") == 1


def test_poll_failure_timeout(prepared: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """连续轮询故障超过预算后 Batch 确实失败。"""
    plc, app = prepared
    app.config = replace(app.config, plc=replace(app.config.plc, comm_fail_timeout_s=.03))

    def failed() -> tuple[int, int]:
        """模拟持续断线。"""
        raise TimeoutError("持续断线")

    monkeypatch.setattr(app.plc.handshake, "read_request", failed)
    with pytest.raises(GuardError, match="BATCH-FAILED"):
        drive_until(app, lambda: False)


def test_reply_failure_exhausts_attempts(prepared: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """三次写失败后终止批次，避免继续处理后续请求。"""
    plc, app = prepared
    attempts = []

    def failed(task: int, func: int, coordinates: object) -> None:
        """每次提交均模拟断线。"""
        attempts.append(task)
        raise TimeoutError("持续写失败")

    monkeypatch.setattr(app.plc.handshake, "write_reply", failed)
    task = plc.request(1)
    with pytest.raises(GuardError, match="BATCH-FAILED"):
        drive_until(app, lambda: False)
    assert attempts == [task, task, task]
    assert plc.io.read("result_id") == 0


def test_tree_log_and_registered_branch(prepared: tuple, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """新增分发表项即可使功能码 77 执行对应子树。"""
    plc, app = prepared
    monkeypatch.setitem(BRANCHES, 77, lambda ctx: PrepareScan(ctx.path))
    task = plc.request(77)
    drive_until(app, lambda: plc.io.read("result_id") == task)
    assert plc.io.read("reply_func") == 1
    assert "功能码 77" in format_tree(app.handle.batch.tree)
    startup = "\n".join(record.message for record in caplog.get_records("setup"))
    assert "主程序行为树结构" in startup
    assert "RepeatUntil: 一盘直到结束" in startup
    assert "Fallback: 功能码分发" in startup
    assert "Sequence: 功能码 60" in startup


def test_reply_retry_after_lost_ack(prepared: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """首次结果已写入但确认丢失时，重试使用完全相同的应答。"""
    plc, app = prepared
    original = app.plc.handshake.write_reply
    attempts = []

    def write(task: int, func: int, coordinates: object) -> None:
        """第一次完成真实写入后模拟应答丢失。"""
        attempts.append((task, func, coordinates))
        original(task, func, coordinates)
        if len(attempts) == 1:
            raise TimeoutError("提交确认丢失")

    monkeypatch.setattr(app.plc.handshake, "write_reply", write)
    task = plc.request(1)
    drive_until(app, lambda: len(attempts) >= 2)
    assert attempts[0] == attempts[1]
    assert attempts[0][0] == task
    assert plc.io.read("result_id") == task


def test_poll_timeout_config_default_and_validation(tmp_path: Path) -> None:
    """旧配置省略新字段时默认十秒，非法预算在启动时拒绝。"""
    import yaml

    data = yaml.safe_load((ROOT / "config/nest_clear.yaml").read_text())
    del data["plc"]["comm_fail_timeout_s"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(config_path).plc.comm_fail_timeout_s == 10.0
    data["plc"]["comm_fail_timeout_s"] = 0
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(GuardError):
        load_config(config_path)

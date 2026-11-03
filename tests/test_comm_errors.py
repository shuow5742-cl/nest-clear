"""验证通信守卫分类、诊断信息和握手非法数据，供维护者定位故障。"""
from pathlib import Path
import socket
import threading

import pytest

from nest_clear.comm import Handshake, PLCClient, load_contract
from nest_clear.comm.codec import encode_real32
from nest_clear.guard import GuardError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contract():
    """加载真实地址表，确保错误消息引用现场名称。"""
    return load_contract(ROOT / "config/plc_contract.yaml")


def test_connect_refused(contract):
    """连接未监听端口应报告连接失败及完整地址上下文。"""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = PLCClient("127.0.0.1", port, 0.1, contract)
    with pytest.raises(GuardError) as captured:
        client.read("task_id")
    assert captured.value.code == "COMM-CONNECT"
    for text in ("127.0.0.1", str(port), "task_id", "40903", "902", "0.1", "原始"):
        assert text in str(captured.value)
    client.close()


@pytest.mark.parametrize("method,exception,expected", [
    ("read", TimeoutError("模拟超时"), "COMM-TIMEOUT"),
    ("read", OSError("模拟读取错误"), "COMM-READ"),
    ("write", OSError("模拟写入错误"), "COMM-WRITE"),
])
def test_transport_errors(contract, method, exception, expected):
    """注入传输故障，验证超时和其他读写错误分别报告。"""
    class Transport:
        connected = True

        def read_holding_registers(self, **kwargs):
            raise exception

        def write_register(self, **kwargs):
            raise exception

        def close(self):
            self.connected = False

    client = PLCClient("127.0.0.1", 502, 0.2, contract)
    client._client = Transport()
    with pytest.raises(GuardError) as captured:
        if method == "read":
            client.read("task_id")
        else:
            client.write("result_id", 1)
    assert captured.value.code == expected
    assert "模拟" in str(captured.value)
    assert "寄存器偏移" in str(captured.value)


def test_plc_exception(contract):
    """PLC 明确返回协议异常码时，不应误认为网络错误。"""
    class Response:
        exception_code = 2

        def isError(self):
            return True

    class Transport:
        connected = True

        def read_holding_registers(self, **kwargs):
            return Response()

    client = PLCClient("127.0.0.1", 502, 0.2, contract)
    client._client = Transport()
    with pytest.raises(GuardError) as captured:
        client.read("task_id")
    assert captured.value.code == "COMM-PLC-EXCEPTION"
    assert "地址表" in captured.value.hint


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_float_and_handshake_context(contract, value):
    """非法浮点数不得写入 PLC，握手错误要保留任务和坐标上下文。"""
    with pytest.raises(GuardError) as codec_error:
        encode_real32(value)
    assert codec_error.value.code == "CODEC-REAL32"
    handshake = Handshake(object(), contract)
    with pytest.raises(GuardError) as captured:
        handshake.write_reply(12, 3, {"pick1_xyz": [1, value, 3]})
    assert captured.value.code == "HANDSHAKE-INVALID"
    for text in ("12", "3", "pick1_xyz", repr(value)):
        assert text in str(captured.value)


def test_real_timeout(contract):
    """真实 TCP 服务接受连接但不应答，验证 pymodbus 超时分类。"""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        done = threading.Event()

        def stall():
            """保持连接不发响应，直到测试完成，模拟 PLC 无响应。"""
            connection, _ = server.accept()
            with connection:
                done.wait(3)

        thread = threading.Thread(target=stall)
        thread.start()
        client = PLCClient("127.0.0.1", port, 0.1, contract)
        try:
            with pytest.raises(GuardError) as captured:
                client.read("task_id")
            assert captured.value.code == "COMM-TIMEOUT"
        finally:
            client.close()
            done.set()
            thread.join(3)

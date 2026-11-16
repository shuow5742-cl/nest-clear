"""用内存应答检查模拟 PLC 的工作请求序列。"""
import pytest

from sim.fake_plc import FakePLC


@pytest.mark.parametrize("terminal", [9, 10])
def test_drive_tray_requests(terminal: int) -> None:
    """模拟扫描、挑取、再扫描和两种终态，不启动服务器。"""
    fake = FakePLC()
    requests = []
    arrivals = []
    replies = iter([
        {"func_code": 1, "scan_xyz": (1, 2, 3)},
        {"func_code": 3},
        {"func_code": 1, "scan_xyz": (4, 5, 6)},
        {"func_code": terminal},
    ])

    def request(code: int) -> int:
        """记录功能码并生成递增的任务编号。"""
        requests.append(code)
        return len(requests)

    def reply(task: int, timeout: float) -> dict:
        """提供预设的 PC 应答。"""
        return next(replies)

    fake.request = request
    fake.wait_reply = reply
    fake.set_actual_xyz = arrivals.append
    result = fake.drive_tray()
    assert requests == [1, 20, 60, 20]
    assert arrivals == [(1, 2, 3), (4, 5, 6)]
    assert [row["func_code"] for row in result] == [1, 3, 1, terminal]


def test_drive_tray_injected_error() -> None:
    """注入错误功能码后，收到 999 即结束。"""
    fake = FakePLC()
    requests = []
    fake.request = lambda code: requests.append(code) or 1
    fake.wait_reply = lambda task, timeout: {"func_code": 999}
    records = fake.drive_tray(injected_func=123)
    assert requests == [123]
    assert records[0]["func_code"] == 999

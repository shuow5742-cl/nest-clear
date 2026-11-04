"""可编程 Modbus TCP 模拟 PLC，为握手及后续整盘测试提供设备行为。"""
import asyncio
from pathlib import Path
import threading
import time
from typing import Any, Sequence

from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import ModbusTcpServer

from nest_clear.comm import PLCClient, load_contract
from nest_clear.comm.codec import encode_real32, encode_u16


class FakePLC:
    """运行真实 TCP 服务器，PLC 内部状态直接写入服务端数据区。"""

    def __init__(
        self, port: int = 0, host: str = "127.0.0.1", wrap: int = 65535,
        contract_path: str | Path | None = None,
    ) -> None:
        """保存服务器及任务回绕参数；端口零由系统分配空闲端口。"""
        if type(wrap) is not int or not 1 <= wrap <= 65535:
            raise ValueError("回绕上限必须在 1 至 65535 之间")
        self.host = host
        self.port = port
        self.wrap = wrap
        self.task_id = 0
        path = contract_path or Path(__file__).resolve().parents[1] / "config/plc_contract.yaml"
        self.contract = load_contract(path)
        self.writes: list[tuple[int, list[int]]] = []
        self._ready = threading.Event()
        self._error: Exception | None = None

    def _trace(self, sending: bool, pdu: Any) -> Any:
        """记录服务器收到的写请求，供测试检查真实 TCP 写入顺序。"""
        if not sending:
            if pdu.function_code in (6, 16):
                self.writes.append((pdu.address + 40001, list(pdu.registers)))
            elif pdu.function_code == 22:
                self.writes.append((pdu.address + 40001, [pdu.and_mask, pdu.or_mask]))
            elif pdu.function_code == 23:
                self.writes.append((pdu.write_address + 40001, list(pdu.write_registers)))
            elif pdu.function_code in (5, 15):
                self.writes.append((pdu.address + 1, list(pdu.bits)))
        return pdu

    def start(self) -> "FakePLC":
        """在后台线程启动服务器，等待监听成功后连接模拟侧客户端。"""
        def run() -> None:
            """为服务器线程创建独立事件循环，供异步服务运行。"""
            async def serve() -> None:
                """建立寄存器数据区并监听，启动失败通知主线程。"""
                try:
                    block = ModbusSequentialDataBlock(1, [0] * 1000)
                    device = ModbusDeviceContext(hr=block)
                    context = ModbusServerContext(devices=device, single=True)
                    self._server = ModbusTcpServer(context, address=(self.host, self.port), trace_pdu=self._trace)
                    await self._server.serve_forever(background=True)
                    self.port = self._server.transport.sockets[0].getsockname()[1]
                    self._ready.set()
                    await self._server.serving
                except Exception as exc:
                    self._error = exc
                    self._ready.set()

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(serve())
            self._loop.close()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise TimeoutError("模拟 PLC 启动超时")
        if self._error:
            raise RuntimeError("模拟 PLC 启动失败") from self._error
        self.io = PLCClient(self.host, self.port, 1, self.contract)
        return self

    def close(self) -> None:
        """关闭客户端和服务器，避免测试之间残留监听端口。"""
        self.io.close()
        asyncio.run_coroutine_threadsafe(self._server.shutdown(), self._loop).result(5)
        self._thread.join(5)

    def __enter__(self) -> "FakePLC":
        """支持上下文管理，进入时启动模拟服务器。"""
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """退出上下文时关闭服务器，即使测试中途失败也释放端口。"""
        self.close()

    def set_status(self, value: int) -> None:
        """设置设备状态字，供后续状态机测试模拟 PLC 状态。"""
        self._set_registers(self.contract["status"].address, [encode_u16(value)])

    def _set_registers(self, address: int, values: list[int]) -> None:
        """在服务器事件循环中更新内部寄存器，不伪装成外部客户端写入。"""
        runtime = self._server.context.devices[0]
        result = asyncio.run_coroutine_threadsafe(
            runtime.async_setValues(16, address - 40001, values), self._loop
        ).result(5)
        if result is not None:
            raise ValueError(f"模拟 PLC 内部写入失败：{result}")

    def set_actual_xyz(self, xyz: Sequence[float]) -> None:
        """设置三个实时反馈坐标，模拟设备到达扫描位置。"""
        if len(xyz) != 3:
            raise ValueError("实时坐标必须有三个轴")
        words = [word for value in xyz for word in encode_real32(value)]
        self._set_registers(self.contract["actual_x"].address, words)

    def request(self, func: int) -> int:
        """先写功能码再递增任务编号，让编号变化成为请求信号。"""
        self.task_id = self.task_id % self.wrap + 1
        self._set_registers(self.contract["request_func"].address, [encode_u16(func)])
        self._set_registers(self.contract["task_id"].address, [encode_u16(self.task_id)])
        return self.task_id

    def wait_reply(self, task_id: int, timeout_s: float = 3) -> dict[str, Any]:
        """等待匹配的提交编号，再读取功能码及所有坐标区域。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.io.read("result_id") == task_id:
                result: dict[str, Any] = {"task_id": task_id, "func_code": self.io.read("reply_func")}
                for name in ("scan", "pick1", "pick2", "calibration1", "calibration2"):
                    result[name + "_xyz"] = self.io.read_real32_block(self.contract[name + "_x"].address, 3)
                return result
            time.sleep(0.01)
        raise TimeoutError(f"等待任务 {task_id} 应答超时")

    def start_tool_calibration(
        self, tool: int, actual_xyz: Sequence[float], timeout_s: float = 3,
        position_error: Sequence[float] = (0, 0, 0),
    ) -> list[dict[str, Any]]:
        """驱动对刀至结束；每次偏移可注入固定的定位误差。"""
        if tool not in (1, 2) or len(position_error) != 3:
            raise ValueError("工具号或定位误差无效")
        self.set_actual_xyz(actual_xyz)
        request_func = 100 if tool == 1 else 101
        records = []
        while True:
            task_id = self.request(request_func)
            reply = self.wait_reply(task_id, timeout_s)
            records.append({"request_func": request_func, **reply})
            if reply["func_code"] in (111, 121, 131, 999):
                return records
            if reply["func_code"] not in ((100, 110) if tool == 1 else (101, 120)):
                raise ValueError(f"对刀收到未处理的功能码 {reply['func_code']}")
            xyz = reply[f"calibration{tool}_xyz"]
            self.set_actual_xyz([a + b for a, b in zip(xyz, position_error)])
            request_func = 110 if tool == 1 else 120

    def drive_tray(self, timeout_s: float = 3) -> list[dict[str, Any]]:
        """驱动一盘请求；按应答模拟到位或挑取完成，返回交互记录。"""
        records: list[dict[str, Any]] = []
        request_func = 1
        while True:
            task_id = self.request(request_func)
            reply = self.wait_reply(task_id, timeout_s)
            records.append({"request_func": request_func, **reply})
            if reply["func_code"] == 1:
                self.set_actual_xyz(reply["scan_xyz"])
                request_func = 1
            elif reply["func_code"] == 3:
                request_func = 60
            elif reply["func_code"] == 9:
                return records
            else:
                raise ValueError(f"模拟流程收到未处理的功能码 {reply['func_code']}")

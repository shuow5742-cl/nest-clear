"""带超时和断线重连的 Modbus TCP 传输，衔接地址契约与握手层。"""
from collections.abc import Sequence
import logging
import socket
import threading
from typing import Any

from autoweaver.comm.modbus_primitive import RegisterIO
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusIOException

from .codec import decode_real32, encode_real32, encode_u16
from .contract import Contract
from ..guard import GuardError

logger = logging.getLogger("nest_clear.comm")


class PLCClient(RegisterIO):
    """实现框架寄存器接口，串行化 TCP 请求并按契约提供具名访问。"""

    def __init__(self, host: str, port: int, timeout_s: float, contract: Contract) -> None:
        """保存连接参数和契约；延迟连接让调用方按需启动。"""
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.contract = contract
        self._client = ModbusTcpClient(host, port=port, timeout=timeout_s, retries=0)
        self._lock = threading.RLock()
        self._ever_connected = False

    def close(self) -> None:
        """关闭连接并记录断线，供调用方释放套接字。"""
        with self._lock:
            connected = self._client.connected
            self._client.close()
            if connected:
                logger.info("PLC 连接已关闭：%s:%s", self.host, self.port)

    def _name(self, address: int) -> str:
        """按起始地址查找契约名称，使错误可对应现场地址表。"""
        for register in self.contract.registers.values():
            if register.address == address:
                return register.name
        return "未知地址"

    def _error(self, code: str, address: int, operation: str, exc: Any, hint: str) -> GuardError:
        """组装完整网络和寄存器上下文，供各类通信失败复用。"""
        offset = address - 40001
        return GuardError(
            code,
            f"{operation}失败，原始异常或响应：{exc!r}",
            f"PLC {self.host}:{self.port}，地址名 {self._name(address)}，地址号 {address}，寄存器偏移 {offset}，超时 {self.timeout_s} 秒",
            hint,
        )

    def _call(self, address: int, method: str, **kwargs: Any) -> Any:
        """执行一次请求；失败立即抛守卫，下一次调用负责重连。"""
        operation = "读取" if method.startswith("read") else "写入"
        with self._lock:
            if not self._client.connected:
                try:
                    connected = self._client.connect()
                    if not connected:
                        raise ConnectionError("连接被拒绝或 PLC 未响应")
                except Exception as exc:
                    logger.warning("PLC 连接失败：%s:%s，原因：%s", self.host, self.port, exc)
                    raise self._error("COMM-CONNECT", address, operation, exc, "检查网线、PLC 和工控机 IP、端口、网络连通性以及 PLC 是否上电并处于 RUN。") from exc
                if self._ever_connected:
                    logger.info("PLC 重连成功：%s:%s", self.host, self.port)
                else:
                    logger.info("PLC 连接成功：%s:%s", self.host, self.port)
                self._ever_connected = True
            try:
                result = getattr(self._client, method)(address=address - 40001, **kwargs)
                if result is None:
                    raise IOError("PLC 未返回响应")
                if result.isError():
                    # PLC 返回的协议异常有明确异常码，与网络超时分开处理。
                    if getattr(result, "exception_code", 0):
                        raise self._error("COMM-PLC-EXCEPTION", address, operation, f"异常码 {result.exception_code}，响应 {result!r}", "检查 PLC 是否 RUN、寄存器地址表和读写权限是否一致，并核对异常码。")
                    raise ModbusIOException(str(result))
                if operation == "读取" and len(getattr(result, "registers", [])) != kwargs["count"]:
                    raise IOError(f"PLC 响应寄存器数量不符，期望 {kwargs["count"]}，实际 {getattr(result, "registers", None)!r}")
                return result
            except GuardError:
                raise
            except Exception as exc:
                self._client.close()
                logger.warning("PLC 通信中断：%s:%s，原因：%s", self.host, self.port, exc)
                detail = str(exc).lower()
                if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in detail or "timed out" in detail or "no response" in detail:
                    code = "COMM-TIMEOUT"
                    hint = "检查网线、网络延迟、PLC 是否 RUN 和超时设置。"
                else:
                    code = "COMM-READ" if operation == "读取" else "COMM-WRITE"
                    hint = "检查网线、PLC 是否 RUN、地址表和寄存器读写权限。"
                raise self._error(code, address, operation, exc, hint) from exc

    def read_u16(self, register: int) -> int:
        """读取单个无符号寄存器，供框架及握手使用。"""
        return self._call(register, "read_holding_registers", count=1).registers[0]

    def write_u16(self, register: int, value: int) -> None:
        """校验并写入单个寄存器，供框架及握手使用。"""
        self._call(register, "write_register", value=encode_u16(value))

    def read_real32_block(self, start: int, count: int) -> list[float]:
        """读取连续 REAL32 并按 CDAB 解码，供实时坐标使用。"""
        words = self._call(start, "read_holding_registers", count=count * 2).registers
        return [decode_real32(words[index:index + 2]) for index in range(0, len(words), 2)]

    def write_real32_block(self, start: int, values: Sequence[float]) -> None:
        """将连续坐标编码后一次写入，避免一组坐标拆散传输。"""
        words = [word for value in values for word in encode_real32(value)]
        self._call(start, "write_registers", values=words)

    def read(self, name: str) -> int | float:
        """按契约名称读取寄存器，避免调用方使用裸地址。"""
        register = self.contract[name]
        if register.type == "UINT16":
            return self.read_u16(register.address)
        return self.read_real32_block(register.address, 1)[0]

    def write(self, name: str, value: int | float) -> None:
        """按契约名称写入寄存器，供模拟设备和后续调用方使用。"""
        register = self.contract[name]
        if register.type == "UINT16":
            self.write_u16(register.address, value)
        else:
            self.write_real32_block(register.address, [value])

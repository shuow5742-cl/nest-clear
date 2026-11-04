"""以握手原语执行 PLC 请求、实时坐标读取和应答。"""
import logging
from typing import Any

from nest_clear.comm import Handshake
from nest_clear.guard import GuardError
from autoweaver.worker import AsyncRequestWorker

logger = logging.getLogger("nest_clear.workers.plc")


class PlcWorker(AsyncRequestWorker):
    """串行完成一条 PLC 操作；事务同步执行在框架线程池。"""
    name = "plc"

    def __init__(self, handshake: Handshake) -> None:
        """绑定已预检的握手和连接。"""
        super().__init__()
        self.handshake = handshake

    def initialize(self) -> int:
        """清空应答并读取启动基线，遗留非零请求只记录不处理。"""
        io = self.handshake.io
        contract = self.handshake.contract
        io.write_u16(contract["result_id"].address, 0)
        io.write_u16(contract["reply_func"].address, 0)
        baseline = io.read_u16(contract["task_id"].address)
        if baseline:
            logger.warning("启动 Task_ID 基线为 %s，忽略遗留请求", baseline)
        return baseline

    def execute(self, payload: dict[str, Any]) -> Any:
        """按 note 类型调用握手或读取三个连续实时坐标。"""
        operation = payload["operation"]
        if operation == "request":
            return self.handshake.read_request()
        if operation == "actual":
            return self.handshake.io.read_real32_block(self.handshake.contract["actual_x"].address, 3)
        if operation == "reply":
            self.handshake.write_reply(payload["task_id"], payload["func"], payload.get("coordinates"))
            return None
        raise GuardError("PLC-OPERATION", f"未知操作 {operation!r}", "PLC Worker", "检查行为树命令。")

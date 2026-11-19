"""以握手原语执行 PLC 请求、实时坐标读取和应答。"""
import logging
from typing import Any

from nest_clear.comm import Handshake
from nest_clear.guard import GuardError, diagnostic
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
        """按启动基线清理应答区或处理遗留请求。"""
        io = self.handshake.io
        contract = self.handshake.contract
        baseline = io.read_u16(contract["task_id"].address)
        if baseline == 0:
            io.write_u16(contract["result_id"].address, 0)
            io.write_u16(contract["reply_func"].address, 0)
            return baseline
        result = io.read_u16(contract["result_id"].address)
        if baseline == result:
            return baseline
        status = io.read_u16(contract["status"].address)
        if status & ((1 << 6) | (1 << 7)):
            func = io.read_u16(contract["request_func"].address)
            logger.info("PLC 急停或初始化，放弃遗留请求 Task_ID=%s 功能码=%s", baseline, func)
            if self.handshake.events:
                self.handshake.events.emit("request_abandoned", task_id=baseline, request_func=func, reason="PLC 急停或初始化")
        else:
            logger.error("启动时发现未应答 Task_ID=%s，提交 999", baseline)
            self.handshake.write_reply(baseline, 999, failure=diagnostic("启动时发现尚未应答的请求", code="PLC-ORPHAN-REQUEST", where="主程序启动", hint="检查 PLC 请求状态后重新初始化。"))
        return baseline

    def execute(self, payload: dict[str, Any]) -> Any:
        """按 note 类型调用握手或读取三个连续实时坐标。"""
        operation = payload["operation"]
        if operation == "request":
            return self.handshake.read_request()
        if operation == "actual":
            return self.handshake.io.read_real32_block(self.handshake.contract["actual_x"].address, 3)
        if operation == "calibration_final":
            name = f"calibration{payload['tool']}_feedback_x"
            return self.handshake.io.read_real32_block(self.handshake.contract[name].address, 3)
        if operation == "reply":
            try:
                if payload.get("diagnostic") is None:
                    self.handshake.write_reply(payload["task_id"], payload["func"], payload.get("coordinates"))
                else:
                    self.handshake.write_reply(payload["task_id"], payload["func"], payload.get("coordinates"), failure=payload["diagnostic"])
            except GuardError as exc:
                if exc.code != "HANDSHAKE-INVALID":
                    raise
                logger.error("应答数据无效，改回 999：%s", exc)
                if self.handshake.events:
                    self.handshake.events.emit("error", task_id=payload["task_id"], operation="reply_validation", reason=str(exc))
                self.handshake.write_reply(payload["task_id"], 999, failure=diagnostic(exc))
            return self.handshake.last_reply_func
        raise GuardError("PLC-OPERATION", f"未知操作 {operation!r}", "PLC Worker", "检查行为树命令。")

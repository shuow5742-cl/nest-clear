"""实现请求读取和按提交顺序应答，衔接地址契约与寄存器传输。"""
from collections.abc import Mapping
import logging
import time
from typing import Any

from .codec import encode_real32, encode_u16
from .contract import Contract
from .function_codes import REPLY_FUNCTION_NAMES, uses_tool_id
from ..guard import GuardError, diagnostic

logger = logging.getLogger("nest_clear.comm")
SCALAR_NAMES = frozenset(("removable_count", "pc_tool_id"))


class Handshake:
    """提供无业务分支的读请求和提交应答原语，供后续流程调用。"""

    def __init__(self, io: Any, contract: Contract, write_readback: bool = True, events: Any = None) -> None:
        """保存传输接口和契约，让握手不硬编码 PLC 地址。"""
        self.io = io
        self.contract = contract
        self.write_readback = write_readback
        self.events = events
        self.inconsistent_reads = 0
        self._readback_error_task = None
        self._readback_error_diagnostic = None

    def read_request(self) -> tuple[int, int, int | None]:
        """读取任务编号、功能码及该请求的工具 ID；变化检测和日志由调用方负责。"""
        while True:
            task_id = self.io.read_u16(self.contract["task_id"].address)
            func_code = self.io.read_u16(self.contract["request_func"].address)
            tool_id = self.io.read_u16(self.contract["plc_tool_id"].address) if uses_tool_id(func_code) else None
            second = self.io.read_u16(self.contract["task_id"].address)
            if task_id == second:
                self.inconsistent_reads = 0
                return task_id, func_code, tool_id
            self.inconsistent_reads += 1
            if self.events and hasattr(self.events, "plc"):
                self.events.plc("task_id_inconsistent", task_id=task_id, first=task_id, second=second,
                                failures=self.inconsistent_reads)
                self.events.emit("task_id_inconsistent", task_id=task_id, first=task_id, second=second,
                                 failures=self.inconsistent_reads)
            logger.debug("Task_ID 双读不一致 %s 次：%s → %s", self.inconsistent_reads, task_id, second)
            if self.inconsistent_reads % 5 == 0 and self.events:
                self.events.emit("warning", operation="task_id_inconsistent", reason=f"Task_ID 双读连续 {self.inconsistent_reads} 次不一致，{task_id} → {second}；检查 PLC 更新时序")
            if self.inconsistent_reads >= 5:
                raise GuardError("COMM-TASK-ID", f"Task_ID 双读连续 {self.inconsistent_reads} 次不一致：{task_id} → {second}", f"PLC 地址 {self.contract['task_id'].address}", "检查 PLC 请求更新时序。")

    def write_reply(self, task_id: int, func_code: int, payload: Mapping[str, Any] | None = None, *, failure: Mapping[str, str] | None = None) -> None:
        """先验证全部数据，再按数据区、功能码、提交编号的顺序写入。"""
        started = time.monotonic()
        readback_result = "ok" if self.write_readback else "disabled"
        if self._readback_error_task == task_id:
            func_code = 999
            failure = self._readback_error_diagnostic
            payload = {name: (value if name == "pc_tool_id" else 0 if name in SCALAR_NAMES else [0, 0, 0]) for name, value in (payload or {}).items()}
        context = f"Task_ID={task_id!r}，功能码={func_code!r}"
        try:
            encode_u16(task_id)
            encode_u16(func_code)
        except GuardError as exc:
            raise GuardError("HANDSHAKE-INVALID", f"任务编号或功能码无效：{exc.what}", context, "检查 PLC 任务编号和应答功能码是否为 UINT16。") from exc
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise GuardError("HANDSHAKE-INVALID", f"坐标数据应为映射，实际 {payload!r}", context, "按坐标名称传入三个轴的数值。")
        blocks: list[tuple[int, list[float]]] = []
        scalar_blocks: list[tuple[int, int]] = []
        for name, values in payload.items():
            where = f"{context}，坐标名={name!r}"
            if name in SCALAR_NAMES:
                try:
                    encode_u16(values)
                except GuardError as exc:
                    raise GuardError("HANDSHAKE-INVALID", f"UINT16 标量 {name} 无效：{values!r}", where, "检查标量是否为 UINT16。") from exc
                scalar_blocks.append((self.contract[name].address, values))
                continue
            if name not in ("scan_xyz", "pick1_xyz", "pick2_xyz", "calibration1_xyz", "calibration2_xyz"):
                raise GuardError("HANDSHAKE-INVALID", f"未知坐标名，实际值 {values!r}", where, "检查坐标名是否与契约一致。")
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise GuardError("HANDSHAKE-INVALID", f"坐标应包含三个轴，实际 {values!r}", where, "按 X、Y、Z 顺序提供三个有限数字。")
            for value in values:
                try:
                    encode_real32(value)
                except GuardError as exc:
                    raise GuardError("HANDSHAKE-INVALID", f"坐标无效，实际值 {values!r}：{exc.what}", where, "检查三个轴是否为有限且可编码的数值。") from exc
            prefix = name[:-4]
            blocks.append((self.contract[f"{prefix}_x"].address, list(values)))
        for address, values in blocks:
            self.io.write_real32_block(address, values)
        for address, value in scalar_blocks:
            self.io.write_u16(address, value)
        self.io.write_u16(self.contract["reply_func"].address, func_code)
        if self.write_readback:
            expected = [(a + i, w) for a, values in blocks for i, w in enumerate(word for v in values for word in encode_real32(v))]
            expected += scalar_blocks + [(self.contract["reply_func"].address, func_code)]
            for address, wanted in expected:
                actual = self.io.read_u16(address)
                if actual != wanted:
                    readback_result = "mismatch"
                    error = GuardError("COMM-READBACK", f"写后回读不一致：期望 {wanted}，实际 {actual}", f"Task_ID={task_id}，PLC 地址 {address}", "检查 PC-PLC 通信、寄存器映射及 PLC 是否覆盖写入值。")
                    if self.events:
                        self.events.emit("error", task_id=task_id, operation="write_readback", reason=str(error))
                        if hasattr(self.events, "plc"):
                            self.events.plc("readback_mismatch", task_id=task_id, address=address,
                                            expected=wanted, actual=actual)
                    logger.error("%s", error)
                    self._readback_error_task = task_id
                    failure = diagnostic(error)
                    self._readback_error_diagnostic = failure
                    for block_address, values in blocks:
                        self.io.write_real32_block(block_address, [0.0] * len(values))
                    for block_address, value in scalar_blocks:
                        echo_address = self.contract["pc_tool_id"].address
                        self.io.write_u16(block_address, value if block_address == echo_address else 0)
                    func_code = 999
                    self.io.write_u16(self.contract["reply_func"].address, func_code)
                    actual_error = self.io.read_u16(self.contract["reply_func"].address)
                    if actual_error != func_code:
                        reason = f"COMM-READBACK Task_ID={task_id} PLC 地址 {self.contract['reply_func'].address} 期望 999 实际 {actual_error}"
                        logger.error("%s", reason)
                        if self.events:
                            self.events.emit("error", task_id=task_id, operation="write_readback", reason=reason)
                    break
        # Result_ID 是 PLC 的提交信号；提前写会使 PLC 读到旧功能码或旧坐标。
        self.io.write_u16(self.contract["result_id"].address, task_id)
        self._readback_error_task = None
        self._readback_error_diagnostic = None
        self.last_reply_func = func_code
        if self.events and getattr(self.events, "status", None):
            try:
                self.events.status.observe({"event": "reply", "task_id": task_id, "reply_func": func_code, "wall": self.events.clock().isoformat()})
            except Exception:
                logger.exception("更新应答状态失败")
        if self.events and func_code in (999, 131, 99, 82, 92):
            details = dict(failure or diagnostic(f"功能码 {func_code} 异常应答", code="REPLY-ERROR", where="应答提交", hint="检查请求及设备状态。"))
            if hasattr(self.events, "reply_diagnostics"):
                self.events.reply_diagnostics[task_id] = details
            self.events.emit("reply_error", task_id=task_id, reply_func=func_code, **details)
        if self.events and hasattr(self.events, "plc"):
            self.events.plc("reply", task_id=task_id, func=func_code, coordinates=dict(payload),
                            removable_count=payload.get("removable_count"), pc_tool_id=payload.get("pc_tool_id"), readback=readback_result,
                            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                            **(details if func_code in (999, 131, 99, 82, 92) else {}))
        logger.info("应答完成：Task_ID=%s，功能码=%s（%s），坐标=%s", task_id, func_code, REPLY_FUNCTION_NAMES.get(func_code, "未知功能码"), dict(payload))

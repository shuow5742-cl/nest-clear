"""实现请求读取和按提交顺序应答，衔接地址契约与寄存器传输。"""
from collections.abc import Mapping
import logging
from typing import Any

from .codec import encode_real32, encode_u16
from .contract import Contract
from .function_codes import REPLY_FUNCTION_NAMES
from ..guard import GuardError

logger = logging.getLogger("nest_clear.comm")


class Handshake:
    """提供无业务分支的读请求和提交应答原语，供后续流程调用。"""

    def __init__(self, io: Any, contract: Contract) -> None:
        """保存传输接口和契约，让握手不硬编码 PLC 地址。"""
        self.io = io
        self.contract = contract

    def read_request(self) -> tuple[int, int]:
        """读取任务编号及功能码；变化检测和日志由调用方负责。"""
        task_id = self.io.read_u16(self.contract["task_id"].address)
        func_code = self.io.read_u16(self.contract["request_func"].address)
        return task_id, func_code

    def write_reply(self, task_id: int, func_code: int, payload: Mapping[str, Any] | None = None) -> None:
        """先验证全部数据，再按数据区、功能码、提交编号的顺序写入。"""
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
        for name, values in payload.items():
            where = f"{context}，坐标名={name!r}"
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
        self.io.write_u16(self.contract["reply_func"].address, func_code)
        # Result_ID 是 PLC 的提交信号；提前写会使 PLC 读到旧功能码或旧坐标。
        self.io.write_u16(self.contract["result_id"].address, task_id)
        logger.info("应答完成：Task_ID=%s，功能码=%s（%s），坐标=%s", task_id, func_code, REPLY_FUNCTION_NAMES.get(func_code, "未知功能码"), dict(payload))

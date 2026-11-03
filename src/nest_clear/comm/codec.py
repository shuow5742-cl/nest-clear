"""寄存器编解码，供客户端和模拟 PLC 使用现场确认的 CDAB 字序。"""
import math
import struct
from typing import Any, Sequence

from ..guard import GuardError


def encode_u16(value: Any) -> int:
    """校验无符号单字，防止非法值截断后写入 PLC。"""
    if type(value) is not int or not 0 <= value <= 65535:
        raise GuardError("CODEC-UINT16", f"UINT16 应为 0 至 65535 的整数，实际 {value!r}", "UINT16 编码", "检查任务编号、功能码或寄存器值。")
    return value


def encode_real32(value: Any) -> list[int]:
    """编码有限浮点数，交换两个字以符合现场 PLC 的 CDAB 字序。"""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("必须是有限数字")
        # IEEE 754 的 ABCD 两字交换为 CDAB；每个字内部字节顺序不变。
        high, low = struct.unpack(">HH", struct.pack(">f", value))
    except (ValueError, TypeError, OverflowError, struct.error) as exc:
        raise GuardError("CODEC-REAL32", f"REAL32 不能编码实际值 {value!r}：{exc}", "REAL32 编码", "检查坐标是否为有限数且处于 REAL32 范围。") from exc
    return [low, high]


def decode_real32(words: Sequence[int]) -> float:
    """将两个 CDAB 寄存器还原为浮点数，供坐标读取使用。"""
    if not isinstance(words, (list, tuple)) or len(words) != 2:
        raise GuardError("CODEC-REAL32", f"REAL32 需要两个寄存器，实际 {words!r}", "REAL32 解码", "检查 PLC 响应长度。")
    try:
        high = encode_u16(words[1])
        low = encode_u16(words[0])
        return struct.unpack(">f", struct.pack(">HH", high, low))[0]
    except GuardError as exc:
        raise GuardError("CODEC-REAL32", f"REAL32 寄存器值无效：{words!r}", "REAL32 解码", "检查 PLC 返回的寄存器值。") from exc

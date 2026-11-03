"""加载 PLC 地址契约，为传输和握手提供名称、类型及方向。"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ..guard import GuardError


@dataclass(frozen=True)
class Register:
    """保存单个逻辑寄存器的契约信息，供通信层按名称查询。"""
    name: str
    address: int
    type: str
    direction: str
    description: str


@dataclass(frozen=True)
class Contract:
    """保存不可变地址表，避免运行时更改通信协议。"""
    registers: Mapping[str, Register]

    def __getitem__(self, name: str) -> Register:
        """按名称取地址；未知名称明确报错，避免写错寄存器。"""
        try:
            return self.registers[name]
        except KeyError:
            raise GuardError("CONTRACT-NAME", f"未声明地址名 {name!r}", "PLC 契约", "检查契约和调用方的地址名称。") from None


def load_contract(path: str | Path) -> Contract:
    """读取并校验契约，阻止类型错误或地址重叠进入通信层。"""
    try:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict) or set(data) != {"registers"} or not isinstance(data["registers"], list):
            raise ValueError("根节点必须包含寄存器列表")
        entries: dict[str, Register] = {}
        occupied: set[int] = set()
        for item in data["registers"]:
            if not isinstance(item, dict) or set(item) != {"name", "address", "type", "direction", "description"}:
                raise ValueError("寄存器字段不完整或存在未知字段")
            register = Register(**item)
            if not isinstance(register.name, str) or not register.name or register.name in entries:
                raise ValueError("寄存器名称为空或重复")
            if type(register.address) is not int or not 40001 <= register.address <= 105536:
                raise ValueError(f"寄存器 {register.name} 的地址无效")
            if register.type not in ("UINT16", "REAL32") or register.direction not in ("PLC→PC", "PC→PLC"):
                raise ValueError(f"寄存器 {register.name} 的类型或方向无效")
            if not isinstance(register.description, str) or not register.description:
                raise ValueError(f"寄存器 {register.name} 的说明无效")
            width = 2 if register.type == "REAL32" else 1
            addresses = set(range(register.address, register.address + width))
            if max(addresses) > 105536 or addresses & occupied:
                raise ValueError(f"寄存器 {register.name} 地址重叠或超出范围")
            occupied.update(addresses)
            entries[register.name] = register
        if not entries:
            raise ValueError("寄存器列表为空")
        return Contract(MappingProxyType(entries))
    except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        raise GuardError("CONTRACT-INVALID", f"契约加载失败：{exc}", str(path), "检查 PLC 契约文件。") from exc

"""将设备状态字转换为纯逻辑决策，供父程序管理主程序。

bit7 初始化中上升沿是唯一启动或重启入口；首次读取即为 1 也算上升沿，
避免父程序在设备初始化期间启动时错过信号。bit6 急停优先于其他动作，
解除急停后不会自动重启，必须等待新的 bit7 上升沿。

按用户裁决，bit4 循环停止与 bit3 循环暂停同义：主程序保持存活但暂停。
暂停是闩锁，暂停位清零不代表恢复，只有 bit2 循环启动才能解除；若 bit2
与 bit3/bit4 同时为 1，暂停赢，避免设备尚未准备好就继续运行。
bit9 信息提示、bit10 一般报警和 bit11 严重故障只供记录，不触发动作。
本模块不读 PLC、不操作文件和进程，供父程序和单元测试共用。
"""
from dataclasses import dataclass

BIT_MANUAL = 0  # 手动模式
BIT_AUTO = 1  # 自动模式
BIT_CYCLE_START = 2  # 循环启动
BIT_PAUSE = 3  # 循环暂停
BIT_STOP = 4  # 循环停止
BIT_POWER_ON = 5  # 上电完成
BIT_ESTOP = 6  # 急停中
BIT_INITIALIZING = 7  # 初始化中
BIT_INIT_DONE = 8  # 初始化完成
BIT_INFO = 9  # 信息提示
BIT_WARN = 10  # 一般报警
BIT_FAULT = 11  # 严重故障

BIT_NAMES = {
    BIT_MANUAL: "手动模式",
    BIT_AUTO: "自动模式",
    BIT_CYCLE_START: "循环启动",
    BIT_PAUSE: "循环暂停",
    BIT_STOP: "循环停止",
    BIT_POWER_ON: "上电完成",
    BIT_ESTOP: "急停中",
    BIT_INITIALIZING: "初始化中",
    BIT_INIT_DONE: "初始化完成",
    BIT_INFO: "信息提示",
    BIT_WARN: "一般报警",
    BIT_FAULT: "严重故障",
}
ACTION_START = "start"
ACTION_RESTART = "restart"
ACTION_ESTOP = "estop"


def bit(word: int, index: int) -> bool:
    """判断指定状态位，供状态机和日志共用。"""
    return bool(word & (1 << index))


def describe(word: int) -> list[str]:
    """列出置位的人话名称，供日志和状态文件使用。"""
    return [name for index, name in BIT_NAMES.items() if bit(word, index)]


def conflicting_bits(word: int) -> bool:
    """识别启动与暂停同时置位，供日志提示协议冲突。"""
    return bit(word, BIT_CYCLE_START) and (
        bit(word, BIT_PAUSE) or bit(word, BIT_STOP)
    )


@dataclass(frozen=True)
class Decision:
    """保存本次动作和暂停闩锁，供进程管理器执行。"""
    action: str | None
    paused: bool
    reason: str = ""


class DeviceStateTracker:
    """记住初始化电平和暂停闩锁，避免持续电平重复启动。"""

    def __init__(self) -> None:
        """初始化边沿及暂停状态，首次置位视为上升沿。"""
        self._initializing = False
        self._paused = False

    def decide(self, word: int, *, child_running: bool) -> Decision:
        """按急停优先、初始化边沿和暂停闩锁生成一次决策。"""
        initializing = bit(word, BIT_INITIALIZING)
        rising = initializing and not self._initializing
        self._initializing = initializing
        if bit(word, BIT_PAUSE) or bit(word, BIT_STOP):
            self._paused = True
        elif bit(word, BIT_CYCLE_START):
            self._paused = False
        if bit(word, BIT_ESTOP):
            return Decision(ACTION_ESTOP if child_running else None, self._paused, "急停中")
        if rising:
            action = ACTION_RESTART if child_running else ACTION_START
            return Decision(action, self._paused, "初始化中上升沿")
        return Decision(None, self._paused)

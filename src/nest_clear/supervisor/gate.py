"""主程序通过文件发送活性点并接收暂停状态。"""
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..paths import main_alive_file, supervisor_state_file

logger = logging.getLogger("nest_clear.supervisor.gate")


class SupervisorGate:
    """提供绝不向主循环抛异常的活性与暂停接口。"""

    def __init__(self, root: str | Path, *, mark: Callable[..., Any] | None = None) -> None:
        """绑定共享目录和可选计时回调，供主程序逐拍调用。"""
        self._alive = main_alive_file(root)
        self._state = supervisor_state_file(root)
        self._mark = mark
        self._paused = False
        self._since = 0.0
        self._next_beat = 0.0

    def beat(self) -> None:
        """每秒写一次活性文件，失败记录后继续主循环。"""
        try:
            now = time.monotonic()
            if now < self._next_beat:
                return
            self._next_beat = now + 1.0
            self._alive.parent.mkdir(parents=True, exist_ok=True)
            self._alive.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            logger.warning("活性打点失败：检查运行目录权限和磁盘空间", exc_info=True)

    def paused(self) -> bool:
        """读取暂停状态并在变化时打计时标记；坏文件保留上次状态。"""
        try:
            try:
                data = json.loads(self._state.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or type(data.get("paused")) is not bool:
                    raise ValueError("paused 应为布尔值")
                value = data["paused"]
            except FileNotFoundError:
                value = False
            if value != self._paused:
                self._paused = value
                now = time.monotonic()
                event = "device_pause_begin" if value else "device_pause_end"
                elapsed = 0.0 if value else now - self._since
                if value:
                    self._since = now
                if self._mark is not None:
                    self._mark(event, paused_s=elapsed)
            return self._paused
        except Exception:
            logger.warning("读取暂停状态或记录标记失败：检查状态文件和回调", exc_info=True)
            return self._paused

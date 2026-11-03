"""按本地日期追加跨进程可关联的事件。"""

from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from nest_clear.paths import daily_file


class EventRecorder:
    """每次写入时选择日期，单行一次写入以支持父子进程并发追加。"""

    def __init__(
        self,
        directory: str | Path,
        process: str = "main",
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """保存根目录、进程身份以及可注入时钟。"""
        self.directory = Path(directory)
        self.process = process
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.monotonic = monotonic
        self.tray_id: str | None = None
        self._lock = threading.Lock()
        self.before_emit: Callable[[dict[str, Any]], None] | None = None
        self.listener: Callable[[dict[str, Any]], None] | None = None

    def emit(self, event: str, **fields: Any) -> None:
        """记录本地时间、单调时间和上下文，通知盘统计器。"""
        now = self.clock()
        if self.process == "supervisor":
            try:
                active = json.loads((self.directory / "tray_active.json").read_text(encoding="utf-8"))
                self.tray_id = active.get("tray_id")
            except (OSError, ValueError, AttributeError):
                self.tray_id = None
        row = {
            "ts": now.isoformat(),
            "mono": self.monotonic(),
            "pid": os.getpid(),
            "process": self.process,
            "tray_id": self.tray_id,
            "event": event,
            **fields,
        }
        if self.before_emit:
            self.before_emit(row)
            row["tray_id"] = self.tray_id
        path = daily_file(self.directory, "events.jsonl", now.date())
        line = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            if self.listener:
                self.listener(row)

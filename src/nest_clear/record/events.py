"""按本地日期追加跨进程可关联的事件。"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from nest_clear.paths import daily_file

logger = logging.getLogger(__name__)


class EventRecorder:
    """每次写入时选择日期，单行一次写入以支持父子进程并发追加。"""

    def __init__(
        self,
        directory: str | Path,
        process: str = "main",
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        machine_id: str = "M01",
        run: bool = False,
    ) -> None:
        """保存根目录、进程身份以及可注入时钟。"""
        self.directory = Path(directory)
        self.process = process
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.monotonic = monotonic
        self.machine_id = machine_id
        started = self.clock()
        self.run_id = started.strftime("%Y%m%d_%H%M%S") if run else None
        self.run_directory = (self.directory / started.strftime("%Y-%m-%d") /
                              f"{started:%H%M%S}_{os.getpid()}") if run else None
        self.seq_event = 0
        self.seq_capture = 0
        self.xid = 0
        self.request_task = None
        self.request_func = None
        self.tray_id: str | None = None
        self._lock = threading.RLock()
        self.before_emit: Callable[[dict[str, Any]], None] | None = None
        self.listener: Callable[[dict[str, Any]], None] | None = None
        self.reply_diagnostics: dict[int, dict[str, str]] = {}
        self.status = None
        if process == "main" and run:
            from nest_clear.ui.status import MainStatus
            self.status = MainStatus(self.directory)

    def emit(self, event: str, **fields: Any) -> None:
        """记录本地时间、单调时间和上下文，通知盘统计器。"""
        try:
            with self._lock:
                self._emit(event, **fields)
        except Exception:
            logger.exception("事件写入失败：%s", event)

    def _emit(self, event: str, **fields: Any) -> None:
        now = self.clock()
        if self.process == "supervisor":
            try:
                active = json.loads((self.directory / "tray_active.json").read_text(encoding="utf-8"))
                self.tray_id = active.get("tray_id")
            except (OSError, ValueError, AttributeError):
                self.tray_id = None
        with self._lock:
            self.seq_event += 1
            sequence = self.seq_event
        if fields.get("task_id") == self.request_task and self.request_task is not None:
            fields.setdefault("request_func", self.request_func)
            fields.setdefault("xid", self.xid)
        row = {
            "ts": now.isoformat(),
            "wall": now.isoformat(), "run_id": self.run_id,
            "machine_id": self.machine_id, "seq_event": sequence,
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
        path = (self.run_directory / "events.jsonl" if self.run_directory else
                daily_file(self.directory, "events.jsonl", now.date()))
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
            if self.status:
                self.status.observe(row)

    def next_capture(self) -> tuple[int, str]:
        with self._lock:
            self.seq_capture += 1
            return self.seq_capture, f"{self.run_id}#{self.seq_capture:06d}"

    def begin_request(self, task_id: int, request_func: int) -> int:
        with self._lock:
            self.xid += 1
            self.request_task, self.request_func = task_id, request_func
            return self.xid

    def meta_update(self, **fields: Any) -> None:
        if self.run_directory is None:
            return
        try:
            with self._lock:
                self.run_directory.mkdir(parents=True, exist_ok=True)
                row = {"run_id": self.run_id, "machine_id": self.machine_id,
                       "wall": self.clock().isoformat(), "mono": self.monotonic(), **fields}
                with (self.run_directory / "meta_updates.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception:
            logger.exception("meta_updates.jsonl 写入失败")

    def plc(self, event: str, **fields: Any) -> None:
        """Append a communication edge without letting recording affect the handshake."""
        if self.run_directory is None:
            return
        try:
            with self._lock:
                now = self.clock()
                self.seq_event += 1
                row = {"event": event, "run_id": self.run_id, "machine_id": self.machine_id,
                       "wall": now.isoformat(), "mono": self.monotonic(), "seq_event": self.seq_event, "xid": self.xid,
                       "task_id": self.request_task, "request_func": self.request_func, **fields}
                self.run_directory.mkdir(parents=True, exist_ok=True)
                with (self.run_directory / "plc.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception:
            logger.exception("PLC 边沿记录失败：%s", event)

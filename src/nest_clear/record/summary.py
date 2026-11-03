"""从盘内事件维护可恢复的摘要。"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from nest_clear.paths import daily_file
from .events import EventRecorder


class TraySummary:
    """每次业务事件后保存检查点，异常退出后可补写中断摘要。"""

    def __init__(self, events: EventRecorder) -> None:
        """恢复未完成盘，随后等待首个请求再创建新盘。"""
        self.events = events
        self.checkpoint = events.directory / "tray_active.json"
        self.data: dict[str, Any] | None = None
        if self.checkpoint.exists():
            try:
                self.data = json.loads(self.checkpoint.read_text(encoding="utf-8"))
                self.finish("interrupted", self.data.get("last_ts", self.data["start_time"]), None)
            except (OSError, ValueError, KeyError) as exc:
                import logging

                logging.getLogger("nest_clear.record.summary").warning("恢复中断盘失败：%s", exc)
        events.before_emit = self.before
        events.listener = self.observe

    def start(self, row: dict[str, Any]) -> None:
        """按当天已有摘要和事件分配盘号。"""
        day = row["ts"][:10]
        path = daily_file(self.events.directory, "events.jsonl", datetime.fromisoformat(row["ts"]).date())
        numbers = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    tray = json.loads(line).get("tray_id")
                    if tray and tray.startswith(day + "_"):
                        numbers.append(int(tray.rsplit("_", 1)[1]))
                except (ValueError, AttributeError):
                    continue
        tray_id = f"{day}_{max(numbers, default=0) + 1:04d}"
        self.events.tray_id = tray_id
        self.data = {
            "tray_id": tray_id,
            "start_time": row["ts"],
            "start_mono": row["mono"],
            "scans": 0,
            "targets": 0,
            "picks": {"1": 0, "2": 0},
            "errors_999": 0,
            "reasons_999": {},
            "pause_count": 0,
            "pause_s": 0.0,
            "pause_start": None,
            "last_error": "",
            "images_dropped": 0,
        }

    def before(self, row: dict[str, Any]) -> None:
        """在首个请求写入前分配盘号。"""
        if row["event"] == "request" and self.data is None:
            self.start(row)

    def observe(self, row: dict[str, Any]) -> None:
        """消费握手、检测及暂停事件，完成时落一行摘要。"""
        event = row["event"]
        if event == "request" and self.data is None:
            self.start(row)
        if self.data is None:
            return
        d = self.data
        d["last_mono"] = row["mono"]
        d["last_ts"] = row["ts"]
        if event == "detection":
            d["scans"] += 1
            d["targets"] += row["count"]
        elif event == "image_dropped":
            d["images_dropped"] = d.get("images_dropped", 0) + 1
        elif event == "error":
            d["last_error"] = row.get("reason", "未知")
        elif event == "reply" and row.get("func") == 3:
            d["picks"][str(row["tool"])] += 1
        elif event == "reply" and row.get("func") == 999:
            d["errors_999"] += 1
            reason = d["last_error"] or "未知"
            d["reasons_999"][reason] = d["reasons_999"].get(reason, 0) + 1
        elif event == "device_pause_begin":
            d["pause_count"] += 1
            d["pause_start"] = row["mono"]
        elif event == "device_pause_end" and d["pause_start"] is not None:
            d["pause_s"] += max(0, row["mono"] - d["pause_start"])
            d["pause_start"] = None
        if event == "tray_end":
            self.finish("normal", row["ts"], row["mono"])
        else:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = self.checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, self.checkpoint)

    def finish(self, status: str, timestamp: str, mono: float | None) -> None:
        """写入结束摘要；恢复时使用最近一次检查点避免跨进程单调时钟误算。"""
        if self.data is None:
            return
        d = self.data
        end_mono = mono if mono is not None else d.get("last_mono", d["start_mono"])
        pause = d["pause_s"]
        if d["pause_start"] is not None:
            pause += max(0, end_mono - d["pause_start"])
        duration = max(0, end_mono - d["start_mono"])
        row = {
            **d,
            "end_time": timestamp,
            "status": status,
            "pause_s": pause,
            "duration_s": duration,
            "net_s": max(0, duration - pause),
        }
        path = daily_file(
            self.events.directory, "summary.jsonl", datetime.fromisoformat(d["start_time"]).date()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.checkpoint.unlink(missing_ok=True)
        self.events.tray_id = None
        self.data = None

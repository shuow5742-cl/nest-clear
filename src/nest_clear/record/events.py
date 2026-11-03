"""按日期写入最小 JSONL 事件记录。"""
from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any


class EventRecorder:
    """将事件追加到调用方指定根目录，不依赖路径模块。"""
    def __init__(self, directory: str | Path) -> None:
        """保存记录根目录，允许测试使用临时目录。"""
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        """附加本地时间并原子串行写入一行 JSON。"""
        now = datetime.now().astimezone()
        path = self.directory / now.strftime("%Y-%m-%d") / "events.jsonl"
        line = json.dumps({"time": now.isoformat(), "event": event, **fields}, ensure_ascii=False, allow_nan=False)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

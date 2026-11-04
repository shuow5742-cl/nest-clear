"""按日期分目录的控制台和文件日志。"""

from datetime import date
import logging
from pathlib import Path
from collections.abc import Callable

from nest_clear.paths import daily_file


class DailyHandler(logging.Handler):
    """每条日志检查本地日期，午夜后切换文件。"""

    def __init__(self, root: str | Path, process: str, *, clock: Callable[[], date] = date.today) -> None:
        """保存记录目录和进程文件名。"""
        super().__init__()
        self.root = root
        self.process = process
        self.clock = clock
        self.day: date | None = None
        self.handler: logging.FileHandler | None = None

    def emit(self, record: logging.LogRecord) -> None:
        """在日期变化时关闭旧文件并打开新文件。"""
        try:
            day = self.clock()
            if self.handler is None or self.day != day:
                if self.handler:
                    self.handler.close()
                path = daily_file(self.root, self.process + ".log", day)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.handler = logging.FileHandler(path, encoding="utf-8")
                self.handler.setFormatter(self.formatter)
                self.day = day
            self.handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """关闭当前日志文件。"""
        if self.handler:
            self.handler.close()
        super().close()


def configure(root: str | Path, process: str) -> None:
    """为入口进程设置控制台和每日文件输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), DailyHandler(root, process)],
    )

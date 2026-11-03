"""集中约定父子进程共享文件及按天归档的日志位置。"""
from datetime import date
from pathlib import Path


def supervisor_state_file(root: str | Path) -> Path:
    """返回父程序状态文件位置，供主程序读取暂停状态。"""
    return Path(root) / "supervisor.json"


def main_alive_file(root: str | Path) -> Path:
    """返回主程序活性文件位置，供看门狗检查。"""
    return Path(root) / "main.alive"


def supervisor_log_file(root: str | Path, day: date | None = None) -> Path:
    """返回当天日志位置，供日志处理器跨日切换目录。"""
    return Path(root) / (day or date.today()).isoformat() / "supervisor.log"

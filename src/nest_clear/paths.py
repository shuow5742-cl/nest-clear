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


def daily_file(root: str | Path, name: str, day: date | None = None) -> Path:
    """返回指定日期的记录文件，供日志、事件与摘要共用。"""
    return Path(root) / (day or date.today()).isoformat() / name


def main_status_file(root: str | Path) -> Path:
    return Path(root) / 'state' / 'main_status.json'


def ui_command_file(root: str | Path) -> Path:
    return Path(root) / 'state' / 'ui_command.json'


def tooltip_preview_file(root: str | Path) -> Path:
    """返回对刀准备模式高频实时预览指标文件路径。"""
    return Path(root) / 'state' / 'tooltip_preview.json'


def live_image_file(root: str | Path, name: str = 'latest') -> Path:
    allowed = {
        'latest', 'preview_latest', 'camcal_latest', 'tooloffset_latest', 'tooloffset_crop',
        'tooloffset_front_latest', 'tooloffset_side_latest',
        'tooloffset_tri_1', 'tooloffset_tri_2', 'tooloffset_tri_3', 'tooloffset_tri_4', 'tooloffset_tri_5',
        'pickacc_latest', 'pickacc_crop', 'calib_front_latest', 'calib_side_latest', 'baseline_front', 'baseline_side',
    }
    if name not in allowed:
        raise ValueError('未知实时图片')
    return Path(root) / 'live' / (name + '.jpg')

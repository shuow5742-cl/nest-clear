"""从事件、日志和盘摘要生成现场可读的中文日报。"""

import argparse
from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any

from nest_clear.config import load_config
from nest_clear.paths import daily_file

LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)[,.]\d+ (ERROR|WARNING) (\S+) (.*)$")


def _rows(path: Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL，跳过崩溃时可能留下的不完整末行。"""
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def _time(value: str) -> str:
    """把带时区的事件时间转换为记录所在本地的时分秒。"""
    try:
        return datetime.fromisoformat(value).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "时间未知"


def _log_groups(root: str | Path, day: date) -> tuple[Counter, list[str]]:
    """按级别、日志器及首句统计，并保留每条 ERROR 给人工清单。"""
    groups: Counter = Counter()
    errors = []
    for name in ("main.log", "supervisor.log"):
        path = daily_file(root, name, day)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = LOG_LINE.match(line)
            if not match:
                continue
            _, clock, level, logger, message = match.groups()
            sentence = re.split(r"[。！？；;]", message, maxsplit=1)[0]
            groups[(level, logger, sentence)] += 1
            if level == "ERROR":
                errors.append(f"{clock} ERROR {logger} {message}")
    return groups, sorted(errors)


def report(root: str | Path, day: date) -> str:
    """汇总一日盘记录、状态变化和需要人工处理的事项。"""
    events = _rows(daily_file(root, "events.jsonl", day))
    trays = _rows(daily_file(root, "summary.jsonl", day))
    lines = [f"设备日报 {day.isoformat()}"]
    processes = [row for row in events if row.get("event") in ("process_start", "process_stop")]
    periods = []
    for row in processes:
        action = "启动" if row["event"] == "process_start" else "停止"
        periods.append(f"{_time(row.get('ts', ''))} {row.get('process')} {action}")
    lines.append("开机时段：" + ("；".join(periods) or "无记录"))
    lines.append(f"盘数：{len(trays)}")
    attention = []
    for tray in trays:
        picks = tray.get("picks", {})
        status = "正常结束" if tray.get("status") == "normal" else "中断"
        tray_id = tray.get("tray_id")
        lines.append(
            f"盘 {tray_id}：{status}，耗时 {tray.get('duration_s', 0):.1f} 秒，"
            f"净耗时 {tray.get('net_s', 0):.1f} 秒，扫描位 {tray.get('scans', 0)}，"
            f"目标 {tray.get('targets', 0)}，挑取 1号 {picks.get('1', 0)} 次、"
            f"2号 {picks.get('2', 0)} 次，999 {tray.get('errors_999', 0)} 次，"
            f"丢图 {tray.get('images_dropped', 0)} 张"
        )
        if status == "中断":
            attention.append(f"{_time(tray.get('end_time', ''))} 中断的盘 {tray_id}")
    replies = []
    reasons: dict[tuple[Any, Any], str] = {}
    for row in events:
        event = row.get("event")
        key = (row.get("tray_id"), row.get("task_id"))
        clock = _time(row.get("ts", ""))
        if event == "error":
            reasons[key] = row.get("reason", "未知")
        if (event == "reply" and row.get("func") == 999) or event == "supervisor_reply_999":
            reason = row.get("reason") or reasons.get(key, "未知")
            label = "代答 999" if event == "supervisor_reply_999" else "999"
            text = f"{clock} {label} 盘 {row.get('tray_id')} Task_ID={row.get('task_id')} 原因：{reason}"
            replies.append(text)
            attention.append(text)
        if event == "watchdog":
            attention.append(f"{clock} 看门狗：{row.get('reason', '未知')}")
        if event == "device_state" and "急停中" in row.get("bits", []):
            if "急停中" not in row.get("previous_bits", []):
                attention.append(f"{clock} 急停中")
    lines.append(f"999 次数：{len(replies)}")
    lines.extend(f"  {item}" for item in replies)
    groups, log_errors = _log_groups(root, day)
    for level in ("ERROR", "WARNING"):
        entries = [(key, count) for key, count in sorted(groups.items()) if key[0] == level]
        lines.append(f"{level}：{sum(count for _, count in entries)} 条")
        for (_, logger, sentence), count in entries:
            lines.append(f"  {logger}：{sentence}（{count} 条）")
    attention.extend(log_errors)
    lines.append("设备状态变化时间线（包含急停及暂停）：")
    for row in events:
        event = row.get("event")
        clock = _time(row.get("ts", ""))
        if event == "device_state":
            lines.append(f"  {clock} {'、'.join(row.get('bits', [])) or '无置位'}")
        if event in ("device_pause_begin", "device_pause_end"):
            action = "暂停开始" if event.endswith("begin") else "暂停结束"
            lines.append(f"  {clock} {action}")
    lines.append("需要人看的：")
    lines.extend(f"  {item}" for item in sorted(attention))
    if not attention:
        lines.append("  无")
    return "\n".join(lines)


def main() -> None:
    """使用业务配置中的记录目录输出指定日期日报。"""
    parser = argparse.ArgumentParser(description="燕窝设备中文日报")
    parser.add_argument("date", nargs="?", default=date.today().isoformat())
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[3] / "config/nest_clear.yaml")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print(report(config.record_directory, date.fromisoformat(args.date)))


if __name__ == "__main__":
    main()

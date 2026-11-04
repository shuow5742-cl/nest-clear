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


def _vector(value: Any) -> str:
    """以毫米精度展示坐标，避免 PLC 浮点噪声干扰阅读。"""
    if value is None:
        return "无"
    return "[" + ", ".join(f"{number:.3f}" for number in value) + "]"


def _failure_lines(reason: Any) -> tuple[str, str]:
    """提取守卫错误的现象和建议，并兼容旧记录中的英文检测诊断。"""
    fields = {}
    raw = str(reason or "未知原因")
    for line in raw.splitlines():
        key, separator, value = line.partition("：")
        if separator and key in ("错误码", "发生了什么", "排查建议"):
            fields[key] = value.strip()
    what = fields.get("发生了什么", raw.splitlines()[0])
    what = what.replace("front 相机", "正面相机").replace("side 相机", "侧面相机")
    what = what.replace("no dark component", "图像中未找到暗色针尖区域")
    what = what.replace("largest component is too small", "图像中的暗色针尖区域过小")
    code = fields.get("错误码", "未知")
    hint = fields.get("排查建议", "检查相关设备和日志。")
    if "图像中未找到暗色针尖区域" in what or "图像中的暗色针尖区域过小" in what:
        hint = "检查背光与曝光，以及针尖是否在画面内。"
    return f"错误码：{code}；发生了什么：{what}", f"排查建议：{hint}"


def _mode_rejection(row: dict[str, Any]) -> str:
    """根据协议解释被拒绝的功能码及恢复操作。"""
    func = row.get("request_func")
    mode = row.get("mode", "未知模式")
    names = {1: "扫描", 60: "挑取完成", 100: "工具1开始对刀", 101: "工具2开始对刀",
             110: "工具1对刀继续", 120: "工具2对刀继续", 130: "对刀完成"}
    if mode == "挑毛中" and func in (100, 101):
        hint = "须先停止挑毛并由 PLC 重新初始化"
    elif func in (110, 120, 130) and mode == "空闲":
        hint = "须先开始对应工具的对刀会话"
    elif mode.startswith("对刀中"):
        hint = "须先完成当前工具对刀，再发送允许的指令"
    else:
        hint = "须先进入允许该指令的工作模式"
    return f"模式不允许：{mode} 时收到 {func}（{names.get(func, '未知指令')}），{hint}"


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
    calibrations = _rows(daily_file(root, "tooltip_summary.jsonl", day))
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
    lines.append("工具标定：")
    for calibration in calibrations:
        lines.append(
            f"  {_time(calibration.get('start_time', ''))} 工具 {calibration.get('tool')}，"
            f"迭代 {calibration.get('iterations', 0)} 次，{calibration.get('result')}"
            f"，"
            f"Δ {_vector(calibration.get('delta'))}，补偿 {_vector(calibration.get('old_compensation'))} → "
            f"{_vector(calibration.get('new_compensation'))}"
        )
        if calibration.get("reason"):
            detail, hint = _failure_lines(calibration["reason"])
            lines.extend((f"    {detail}", f"    {hint}"))
    if not calibrations:
        lines.append("  无")
    rejected = {(row.get("tray_id"), row.get("task_id")) for row in events
                if row.get("event") == "mode_rejected"}
    rejected_logs = {(str(row.get("task_id")), str(row.get("reason", ""))) for row in events
                     if row.get("event") == "mode_rejected" and row.get("task_id") is not None}
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
            if event != "reply" or key not in rejected:
                attention.append(text)
        if event == "tooltip_failed":
            detail, hint = _failure_lines(row.get("reason"))
            attention.append(f"{clock} 对刀 131 工具 {row.get('tool')} Task_ID={row.get('task_id')}\n    {detail}\n    {hint}")
        if event == "mode_rejected":
            reason = row.get("reason", "未知")
            reasons[key] = reason
            attention.append(f"{clock} 模式拒绝 999 Task_ID={row.get('task_id')} {_mode_rejection(row)}")
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
    for error in log_errors:
        match = re.search(r"Task_ID=(\d+) 处理失败：(.*)$", error)
        if match and (match.group(1), match.group(2)) in rejected_logs:
            continue
        attention.append(error)
    lines.append("设备状态变化时间线（包含急停及暂停）：")
    for row in events:
        event = row.get("event")
        clock = _time(row.get("ts", ""))
        if event == "device_state":
            lines.append(f"  {clock} {'、'.join(row.get('bits', [])) or '无置位'}")
        if event == "mode_change":
            lines.append(f"  {clock} 模式 {row.get('previous_mode')} → {row.get('mode')}")
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

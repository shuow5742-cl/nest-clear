"""从事件、日志和盘摘要生成现场可读的中文日报。"""

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import json
import math
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


def _day_rows(root: str | Path, name: str, day: date, timestamp: str) -> list[dict[str, Any]]:
    """读取旧版每日文件及所有运行目录，运行记录按行时间而非目录日期归日。"""
    rows = _rows(daily_file(root, name, day))
    for path in Path(root).glob(f"????-??-??/*/{name}"):
        rows.extend(row for row in _rows(path)
                    if str(row.get(timestamp, row.get("ts", "")))[:10] == day.isoformat())
    return rows


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
    names = {1: "扫描", 60: "挑取完成", 100: "工具1对刀", 110: "工具2对刀", 130: "对刀完成"}
    if mode == "挑毛中" and func in (100, 110):
        hint = "须先停止挑毛并由 PLC 重新初始化"
    elif func == 130 and mode == "空闲":
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
    paths = [daily_file(root, name, day) for name in ("main.log", "supervisor.log")]
    paths.extend(Path(root).glob("????-??-??/*/main.log"))
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = LOG_LINE.match(line)
            if not match:
                continue
            log_day, clock, level, logger, message = match.groups()
            if log_day != day.isoformat():
                continue
            sentence = re.split(r"[。！？；;]", message, maxsplit=1)[0]
            groups[(level, logger, sentence)] += 1
            if level == "ERROR":
                errors.append(f"{clock} ERROR {logger} {message}")
    return groups, sorted(errors)


def _work_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """消费工作事件与明确的 pick_result，不从坐标反推结局。"""
    stats = {"coarse": 0, "fine": 0, "picks": 0, "success": 0, "failed": 0,
             "pending": 0, "not_sent": 0, "fine_failures": Counter(), "limits": [], "checks": [],
             "replies": Counter(), "rejections": Counter(), "missing": 0, "failed_targets": [], "ambiguous": 0}
    out_ids = set()
    out_unidentified = 0
    captures = {r.get("capture_id"): r for r in rows if r.get("event") == "capture"}
    latest_outcomes = {r.get("attempt_id"): r for r in rows
                       if r.get("event") == "pick_result" and r.get("attempt_id")}
    for row in rows:
        event = row.get("event")
        if event == "pick_result" and row.get("outcome") in ("success", "failed", "pending", "not_sent"):
            if row.get("attempt_id") and latest_outcomes.get(row["attempt_id"]) is not row:
                continue
            stats[row["outcome"]] += 1
            if row["outcome"] == "failed":
                first = row.get("coarse_capture_id") or row.get("capture_id")
                names = [Path(captures.get(key, {}).get("raw_path") or "").name
                         for key in (first, row.get("review_capture_id"))]
                stats["failed_targets"].append((row.get("target_id"), names))
        elif event == "detection":
            stats["rejections"].update(row.get("rejections", {}))
            if row.get("kind") == "coarse":
                for candidate in row.get("candidates", []):
                    if candidate.get("status") == "out_of_range":
                        if candidate.get("target_id"):
                            out_ids.add(candidate["target_id"])
                        else:
                            out_unidentified += 1
        elif event == "image_dropped":
            stats["missing"] += 1
        elif event == "match_ambiguous":
            stats["ambiguous"] += 1
        elif event == "reply_sent":
            stats["picks"] += 1
        if row.get("event") == "reply" and row.get("func") in (60, 9, 10, 999):
            stats["replies"][row["func"]] += 1
        if row.get("event") != "work_cycle":
            continue
        kind = row.get("type")
        if kind == "fine":
            stats["fine"] += 1
            if row.get("reason") != "ok":
                stats["fine_failures"][row.get("reason")] += 1
        elif kind == "coarse":
            stats["coarse"] += 1
            if row.get("reason") in ("cap", "coarse_limit"):
                stats["limits"].append((row.get("view_index"), row.get("reason")))
            if row.get("reason") in ("empty", "cap", "coarse_limit"):
                stats["checks"].append({key: row.get(key) for key in ("view_index", "tool_change", "attempts", "tool_check_min_attempts", "tool_check_total", "tool_check_threshold")})
    stats["out_of_range"] = len(out_ids) + out_unidentified
    return stats


def report(root: str | Path, day: date) -> str:
    """汇总一日盘记录、状态变化和需要人工处理的事项。"""
    events = _day_rows(root, "events.jsonl", day, "wall")
    trays = _day_rows(root, "summary.jsonl", day, "start_time")
    calibrations = _day_rows(root, "tooltip_summary.jsonl", day, "end_time")
    # 统计严格按事件发生日期归日；跨午夜的运行不需要拆分文件。
    tray_events = list(events)
    lines = [f"设备日报 {day.isoformat()}"]
    processes = [row for row in events if row.get("event") in ("process_start", "process_stop")]
    periods = []
    for row in processes:
        action = "启动" if row["event"] == "process_start" else "停止"
        periods.append(f"{_time(row.get('ts', ''))} {row.get('process')} {action}")
    lines.append("开机时段：" + ("；".join(periods) or "无记录"))
    lines.append(f"盘数：{len(trays)}")
    arrivals = [r for r in events if r.get("event") in ("arrival", "arrival_warning") and "stable" in r]
    deviations = [(abs(float(d)), axis) for r in arrivals for axis, d in zip("XYZ", r.get("delta", []))]
    durations = sorted(float(r["settle_elapsed_s"]) for r in arrivals if r.get("settle_elapsed_s") is not None)
    from statistics import median
    maximum = max(deviations, default=None)
    lines.append("到位监督：超差次数 {}，最大偏差 {}，未稳定次数 {}，稳定耗时中位数/最大 {}".format(
        sum(bool(r.get("over_tolerance")) for r in arrivals),
        f"{maximum[0]:.3f} mm ({maximum[1]}轴)" if maximum else "无",
        sum(not r["stable"] for r in arrivals),
        f"{median(durations):.3f}/{max(durations):.3f} 秒" if durations else "无"))
    preparations = [row for row in events if row.get('event') == 'tooltip_preparation_finished']
    latest = (preparations[-1].get('cameras') or {}) if preparations else {}
    widths = ' / '.join(f"{label} {latest.get(name, {}).get('edge_width_um', '未测')} µm" for name, label in (('front','主'), ('side','辅')))
    lines.append(f"对刀相机准备：{len(preparations)} 次，最终边缘宽度 {widths}")
    attention = []
    tray_ids = list(dict.fromkeys([row.get("tray_id") for row in trays] +
                                  [row.get("tray_id") for row in events if row.get("tray_id") and row.get("event") == "work_cycle"]))
    for tray_id in tray_ids:
        tray = next((row for row in trays if row.get("tray_id") == tray_id), {})
        stats = _work_stats([row for row in tray_events if row.get("tray_id") == tray_id])
        settled = stats["success"] + stats["failed"]
        rate = f"{stats['success'] / settled:.1%}" if settled else "无"
        status = "正常结束" if tray.get("status") == "normal" else "中断或进行中"
        lines.append(f"盘 {tray_id}：{status}，净耗时 {tray.get('net_s', 0):.1f} 秒，粗拍 {stats['coarse']} 次，精定位 {stats['fine']} 次，"
                     f"精定位失败 {dict(stats['fine_failures'])}，挑取 {stats['picks']} 次，"
                     f"成功 {stats['success']}、失败 {stats['failed']}、待判定 {stats['pending']}，"
                     f"未下发 {stats['not_sent']}，范围外 {stats['out_of_range']} 根，成功率 {rate}，拒绝原因 {dict(stats['rejections'])}，缺图 {stats['missing']}，"
                     f"上限视野 {stats['limits']}，回 10 判定 {stats['checks']}，"
                     f"回 60/9/10/999 次数 {stats['replies'].get(60, 0)}/{stats['replies'].get(9, 0)}/{stats['replies'].get(10, 0)}/{stats['replies'].get(999, 0)}")
        for target_id, names in stats["failed_targets"]:
            lines.append(f"  失败目标 {target_id}：{names[0]} → {names[1]}")
        if stats["missing"]:
            attention.append(f"盘 {tray_id} 缺图 {stats['missing']} 张")
        if stats["picks"] and stats["pending"] / stats["picks"] > .2:
            attention.append(f"盘 {tray_id} 待判定超过 20%：{stats['pending']}/{stats['picks']}")
        if stats["ambiguous"]:
            attention.append(f"盘 {tray_id} 身份匹配歧义 {stats['ambiguous']} 次")
        if tray and tray.get("status") != "normal":
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
        clicks = [c for c in calibration.get("tip_clicks", []) if c.get("source") == "manual"]
        if clicks:
            details = "；".join(f"{c.get('camera')} {c.get('correction_px')} px / {c.get('correction_mm')} mm" for c in clicks)
            lines.append(f"    人工点击：是，修正 {details}")
        else:
            lines.append("    人工点击：否")
        widths = calibration.get("final_edge_width_px")
        if widths:
            def display_width(view: str) -> str:
                value = widths.get(view)
                return "未测" if value is None else f"{value:.1f} px"
            width_text = f"正面 {display_width('front')} / 侧面 {display_width('side')}"
        else:
            width_text = "未记录"
        lines.append(f"    离焦轮数 {calibration.get('defocus_iterations', 0)}，最终边缘宽度 {width_text}")
        if calibration.get("reason"):
            detail, hint = _failure_lines(calibration["reason"])
            lines.extend((f"    {detail}", f"    {hint}"))
    if not calibrations:
        lines.append("  无")
    lines.append("相机标定：")
    camera_rows = [row for row in events if row.get("event") in ("camcal_done", "camcal_failed", "camcal_aborted")]
    for row in camera_rows:
        status = {"camcal_done": "成功", "camcal_failed": "失败", "camcal_aborted": "中断"}[row["event"]]
        lines.append(f"  {_time(row.get('start_time', ''))} → {_time(row.get('end_time', ''))}，{row.get('n_images', 0)} 张，{status}")
        if row["event"] == "camcal_done":
            r = row.get("report") or {}
            lines.append(f"    RMS {r.get('rms_px')} px，主点 {r.get('um_per_px_at_principal')} µm/px，各轴比例误差 {r.get('axis_scale')}，相机倾角 {r.get('camera_tilt_deg')}°")
        else:
            lines.append(f"    {row.get('code', '中断')}：{row.get('reason', '未知')}")
            if row["event"] == "camcal_failed":
                attention.append(f"{_time(row.get('end_time', ''))} 相机标定 82 {row.get('code')}：{row.get('reason')}")
    if not camera_rows:
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
        if event in ("heartbeat_failed", "heartbeat_restored", "placeholder_calibration", "tooltip_quality_warning") or (event == "error" and (row.get("operation") in ("image", "write_readback", "clear_algorithm_ready", "write_131") or "CAMERA-CAPTURE" in str(row.get("reason", "")))):
            labels = {"heartbeat_failed": "心跳中断", "heartbeat_restored": "心跳恢复", "placeholder_calibration": "占位标定放行中", "tooltip_quality_warning": "对刀质量超限", "error": "设备或记录错误"}
            reason = str(row.get("reason") or row).replace("\n", "；")
            attention.append(f"{clock} Task_ID={row.get('task_id')} {labels[event]}：{reason}；建议：检查相关设备、配置与日志。")
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

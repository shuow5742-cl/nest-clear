"""文件和事件到显示数据的转换，不依赖 tkinter。"""
import json
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from nest_clear.display import format_number
from nest_clear.paths import tooltip_preview_file

BITS = ('手动模式','自动模式','循环启动','循环暂停','循环停止','上电完成','急停中','初始化中','初始化完成','信息提示','一般报警','严重故障')
TOOLTIP_PREVIEW_STALE_SECONDS = 3.0


def _now_and_epoch(now=None):
    if now is None:
        dt = datetime.now().astimezone()
        return dt, dt.timestamp()
    if isinstance(now, (int, float)):
        dt = datetime.fromtimestamp(float(now), tz=timezone.utc).astimezone()
        return dt, float(now)
    dt = now if now.tzinfo else now.astimezone()
    return dt, dt.timestamp()


def extract_tooltip_preview(doc, camera, *, now=None, file_age=None, stale_seconds=TOOLTIP_PREVIEW_STALE_SECONDS):
    """提取单台相机的实时预览指标；文件缺失或超过 stale_seconds 秒未更新时返回 None。"""
    if file_age is not None and file_age > stale_seconds:
        return None
    if not isinstance(doc, dict) or not doc:
        return None
    cameras = doc.get('cameras') if isinstance(doc.get('cameras'), dict) else doc
    cam = cameras.get(camera) if isinstance(cameras, dict) else None
    if not isinstance(cam, dict) or not cam:
        return None
    ts = cam.get('updated_at') or cam.get('wall') or cam.get('ts') or doc.get('updated_at')
    if ts is None:
        return cam if file_age is not None else None
    now_dt, now_epoch = _now_and_epoch(now)
    if isinstance(ts, (int, float)):
        if now_epoch - float(ts) > stale_seconds:
            return None
    else:
        dt = parse_time(ts)
        if dt is None or (now_dt - dt).total_seconds() > stale_seconds:
            return None
    return cam


def read_tooltip_preview(root, *, now=None, stale_seconds=TOOLTIP_PREVIEW_STALE_SECONDS):
    """读取 <root>/state/tooltip_preview.json 并过滤掉过期相机；文件缺失或过期返回 {}。"""
    path = tooltip_preview_file(root)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    _, now_epoch = _now_and_epoch(now)
    file_age = max(0.0, now_epoch - mtime)
    if file_age > stale_seconds:
        return {}
    doc = read_json(path)
    result = {}
    for name in ('front', 'side'):
        cam = extract_tooltip_preview(doc, name, now=now, file_age=file_age, stale_seconds=stale_seconds)
        if cam is not None:
            result[name] = cam
    return result


def baseline_preview_display(item, live=None, *, allow_inline=False):
    """Format one preparation view, discarding stale metrics after an error or expiration."""
    error = str(item.get('error') or '')
    if error:
        hint = ('两台相机不要接在同一个 USB 集线器上，换一个主板 USB 口后重开预览。'
                if 'USB' in error or '带宽' in error else '检查相机配置、连接和权限后重开预览。')
        return {'failed': True, 'on': bool(item.get('on')), 'no_live_data': False,
                'button': '预览失败，点击重试', 'message': error, 'hint': hint,
                'metrics': {'percent': '—', 'current': '—', 'peak': '—', 'edge_um': '—',
                            'edge_px': '—', 'contrast': '—', 'source': '—', 'exposure': '—'},
                'history': []}
    if not item.get('on'):
        return {'failed': False, 'on': False, 'no_live_data': False, 'message': '预览关闭',
                'metrics': None, 'history': []}
    src = live if isinstance(live, dict) and live else (item if allow_inline and 'percent' in item else None)
    if not src:
        exposure_error = item.get('exposure_error')
        message = str(exposure_error) if exposure_error else '无实时数据'
        return {'failed': False, 'on': True, 'no_live_data': True, 'message': message,
                'metrics': {'percent': '—', 'current': '—', 'peak': '—', 'edge_um': '—',
                            'edge_px': '—', 'contrast': '—', 'source': '—', 'exposure': '—'},
                'history': []}
    return {'failed': False, 'on': True, 'no_live_data': False, 'live': src,
            'history': list(src.get('history') or [])}


def page_for_mode(status):
    """将主程序状态中的工作模式映射到页面；空闲和未知模式不切页。"""
    mode = status.get('mode') if isinstance(status, dict) else None
    if not isinstance(mode, str): return None
    return {
        '相机标定中': '相机标定',
        '工具偏移标定中': '工具偏移标定',
        '下针深度标定中': '下针深度标定',
        '挑取精度标定中': '挑取精度标定',
        '对刀基准建立中': '对刀基准建立',
        '对刀中(工具1)': '换针对刀',
        '对刀中(工具2)': '换针对刀',
        '挑毛中': '主页',
    }.get(mode)


def pickacc_tray_geometry(width, height):
    """行程框等比例居中；箭头按 0.05 mm 约 30 像素选择常用倍率。"""
    scale = max(.01, min((width - 90) / 100, (height - 45) / 160))
    left = (width - 100 * scale) / 2
    bottom = (height + 160 * scale) / 2 - 5
    target = 600 / scale
    decade = 10 ** math.floor(math.log10(target))
    multiplier = min((v * decade for v in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), key=lambda v: abs(v-target))
    return left, bottom, scale, int(multiplier)


def pickacc_type(value):
    return {'constant':'固定补偿', 'table':'补偿表'}.get(value, '未标定' if not value else value)


def pickacc_z_type(value):
    return {'constant': '全盘最大值', 'table': 'Z 表插值'}.get(value, '未标定' if not value else value)


def pickacc_z_summary_text(data):
    """格式化当前点 Z、已测点 Z 列表、当前极差及最终 Z 模式。"""
    samples = data.get('samples') or []
    z_rows = [(s.get('target'), s.get('pick_z_mm')) for s in samples if not s.get('discarded') and isinstance(s.get('pick_z_mm'), (int, float))]
    pending = data.get('pending_sample') or {}
    phase = data.get('phase')
    if phase == 'z_step':
        current_z = f"待测（{pending.get('target') or data.get('target') or '当前点'} 压住 0.05mm 塞尺后按【记录下针 Z】）"
    elif z_rows:
        current_z = f'{z_rows[-1][0]} {z_rows[-1][1]:.3f} mm'
    else:
        current_z = '—'
    jitter = data.get('z_jitter_mm')
    jitter_text = f'{jitter:.3f} mm' if isinstance(jitter, (int, float)) else '—'
    if z_rows:
        zs = [z for _, z in z_rows]
        recent = '、'.join(f'{t}:{z:.3f}' for t, z in z_rows[-8:])
        if len(z_rows) > 8:
            recent = '… ' + recent
        list_text = f'{len(z_rows)} 点（{min(zs):.3f}~{max(zs):.3f}，差 {max(zs)-min(zs):.3f} mm）[{recent}]'
    else:
        list_text = '无'
    candidate = data.get('candidate') or {}
    z_summary = data.get('z_summary') or {}
    z_src = candidate if candidate.get('z_type') else z_summary
    z_type = z_src.get('z_type')
    z_decision = z_src.get('z_decision') or {}
    if z_type == 'constant' and isinstance(z_src.get('z_mm'), (int, float)):
        mode_text = f"全盘最大值 {z_src['z_mm']:.3f} mm（最大差 {z_decision.get('range_mm', 0):.3f} ≤ {z_decision.get('threshold_mm', 0.1):.3f} mm）"
    elif z_type == 'table':
        mode_text = f"Z 表插值（{len(z_src.get('z_nodes') or [])} 点，{z_decision.get('min_z_mm', 0):.3f}~{z_decision.get('max_z_mm', 0):.3f} mm，最大差 {z_decision.get('range_mm', 0):.3f} > {z_decision.get('threshold_mm', 0.1):.3f} mm）"
    else:
        mode_text = '待判定'
    return f'该点 Z：{current_z}　|　当前极差：{jitter_text}　|　已测 Z：{list_text}\n下针 Z 模式：{mode_text}'


def pickacc_z_missing_tools(existing):
    """返回未做下针深度标定的工具编号列表；existing 为空时不报错。"""
    if not isinstance(existing, dict) or not existing:
        return []
    missing = []
    for tool in (1, 2):
        old = existing.get(f'tool{tool}')
        if isinstance(old, dict) and not old.get('z_calibrated') and old.get('z_type') not in ('constant', 'table'):
            missing.append(tool)
    return missing


def pickacc_stage(value):
    return {'A':'第一轮', 'B':'补点', 'repeat':'重复性', 'verify':'验证',
            'infield':'视野内', 'simple':'工具2简化', 'outlier_retest':'离群重测',
            'discard':'作废'}.get(value, value or '—')


def pickacc_note(sample):
    parts = []
    if sample.get('outlier'): parts.append('离群')
    if sample.get('retest'): parts.append('重测')
    if sample.get('verification'): parts.append('验证')
    if sample.get('stage') == 'infield':
        corner = sample.get('corner')
        labels = {(-1,1):'左上', (1,1):'右上', (-1,-1):'左下', (1,-1):'右下'}
        label = labels.get(tuple(corner)) if isinstance(corner, (tuple, list)) else None
        parts.append('视野内' + ('-' + label if label else ''))
    if sample.get('discarded'): parts.append('作废')
    return '、'.join(parts) or '拟合'


def pickacc_infield(edges):
    if not edges: return '视野内：未开始'
    maximum = max(float(edge.get('distance_mm', 0)) for edge in edges)
    state = '⚠ 框内误差不均，相机标定需复核' if any(edge.get('warning') for edge in edges) else ('正常' if len(edges) >= 4 else '检查中')
    return f'视野内：四角最大差 {maximum:.3f} mm {state}'

def decode_status(word):
    if not isinstance(word, int):
        return [{'bit': i, 'name': n, 'active': False, 'color': 'gray'} for i,n in enumerate(BITS)]
    return [{'bit': i, 'name': n, 'active': bool(word & (1 << i)), 'color': ('red' if i in (6,11) else 'yellow' if i == 10 else 'green') if word & (1 << i) else 'gray'} for i,n in enumerate(BITS)]

def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}

def parse_time(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.astimezone()
    except (ValueError, TypeError, OverflowError):
        return None

def elapsed(value, now=None):
    dt = parse_time(value)
    if dt is None: return None
    now = now or datetime.now().astimezone()
    return max(0, int((now - dt).total_seconds()))

def relative_time(value, now=None):
    seconds = elapsed(value, now)
    if seconds is None: return '无数据', 'gray'
    return f'{seconds} 秒前', 'red' if seconds > 30 else 'yellow' if seconds > 10 else 'green'

def display_time(value, today=None):
    dt = parse_time(value)
    if dt is None: return str(value or '无数据')
    today = today or date.today()
    return dt.strftime('%H:%M:%S' if dt.date() == today else '%m-%d %H:%M:%S')

def duration(value, now=None):
    seconds = elapsed(value, now)
    if seconds is None: return '无数据'
    days, seconds = divmod(seconds, 86400); hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    return (f'{days}天 ' if days else '') + f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def picking_text(p):
    if not isinstance(p, dict) or not p: return '无数据'
    def val(*keys, default='—'):
        return next((p[k] for k in keys if p.get(k) is not None), default)
    if p.get('film_mode'):
        return f"菲林测试 · 可去 {val('removable_count')} · 不可去 {val('non_pickable_count')} · 已下发 {val('sent_count')} · 失败 {val('failed_count')}"
    tray = val('tray_id','tray'); view = val('view_index','view'); total = val('total_views','total')
    if isinstance(view, int) and 'view_index' in p: view += 1
    return f"第 {tray} 盘 · 视野 {view}/{total} · 粗拍 {val('coarse_shots')} 次 · 可去 {val('removable_count', 'n0')} · 红框 {val('red_count')} · 已下发 {val('sent_count', 'attempts')}"

# 父程序默认每 0.1 秒发布；30 秒容忍 300 个周期及短暂调度延迟。
SUPERVISOR_STALE_SECONDS = 30

def supervisor_live(s, age):
    return age is not None and age <= SUPERVISOR_STALE_SECONDS and bool(s.get('pid'))


def parent_process_row(s, supervisor_age, now=None):
    live = supervisor_live(s, supervisor_age)
    return ('父程序', s.get('pid') if live else '—', display_time(s.get('start_time'), _now_and_epoch(now)[0].date()),
            duration(s.get('start_time'), now) if live else '—', '运行' if live else '未运行', '—')


def child_process_row(s, supervisor_age, now=None):
    live = supervisor_live(s, supervisor_age)
    pid = s.get('child_pid') if live else None
    return ('主程序', pid or '—', display_time(s.get('child_start_time'), _now_and_epoch(now)[0].date()),
            duration(s.get('child_start_time'), now) if pid else '—',
            (s.get('child_state') or '无数据') if live else '未运行',
            s.get('child_last_exit_reason') or '无')


def status_indicators(s, m, now=None, supervisor_age=None):
    plc_text, plc_color = relative_time(s.get('plc_last_read'), now)
    hb_text, hb_color = relative_time(s.get('heartbeat_last_write'), now)
    if s.get('plc_connected') is False: plc_color = 'red'
    if s.get('heartbeat_interrupted'): hb_color = 'red'
    live = supervisor_live(s, supervisor_age)
    child = live and bool(s.get('child_pid')) and s.get('child_state') in ('启动中', '运行', '就绪')
    main_indicator = (('yellow' if s.get('child_state') == '启动中' else 'green'), str(m.get('mode') or s.get('child_state'))) if child else ('gray', '主程序未运行')
    ready_text = str(s.get('algorithm_ready') if s.get('algorithm_ready') is not None else '无数据')
    if live:
        parent = ('green', str(s['pid']))
        ready = ('green' if s.get('algorithm_ready') == 1 else 'gray', ready_text)
    else:
        parent = ('gray', f'未运行({int(supervisor_age)}秒过期)' if supervisor_age is not None and supervisor_age > SUPERVISOR_STALE_SECONDS else '无数据')
        plc_text, hb_text, ready_text = (text + '(过期)' if text != '无数据' else text for text in (plc_text, hb_text, ready_text))
        plc_color = hb_color = 'gray'
        ready = ('gray', ready_text)
    return {'父程序': parent, '主程序': main_indicator, 'PLC 连接': (plc_color, plc_text), '心跳': (hb_color, hb_text), '40932 就绪': ready}

EVENTS = {'arrival_warning','tooltip_settings_warning','tooltip_preview_error','tooltip_exposure_failed','error','pickacc_failed','tooloffset_failed','tooloffset_quality_warning','camcal_failed','tooltip_failed','supervisor_reply_999','request_abandoned','plc_disconnected','plc_restored','heartbeat_failed','heartbeat_restored','image_dropped','warning'}
def classify_event(row):
    event = str(row.get('event','')).lower()
    text = ' '.join(str(row.get(k,'')) for k in ('reason_code','code','reason','what','operation','event')).lower()
    reply = row.get('reply_func', row.get('func_code'))
    if reply is None and event == 'reply': reply = row.get('func')
    if reply is None and event == 'supervisor_reply_999': reply = 999
    if event not in EVENTS and reply not in (999, 131, 99, 82) and not any(x in text for x in ('readback','回读不一致','采图失败','capture_failed')):
        return None
    category = '父程序' if row.get('process') == 'supervisor' else '标定' if any(x in text for x in ('camcal','tooltip','tooloffset','pickacc','calib')) or reply in (82, 99, 131, 152) else '通信' if any(x in text for x in ('plc','heartbeat','readback','回读')) else '相机' if any(x in text for x in ('camera','capture','image','采图')) else '配置' if 'config' in text else '算法'
    color = 'green' if 'restored' in event or '恢复' in event else 'yellow' if event in ('arrival_warning','warning','tooltip_settings_warning','tooloffset_quality_warning','request_abandoned','image_dropped') else 'red'
    reason = str(row.get('reason',row.get('message',event)))
    fields = {}
    for line in reason.splitlines():
        for prefix, key in (('发生了什么：','what'), ('位置：','where'), ('排查建议：','hint')):
            if line.startswith(prefix): fields[key] = line[len(prefix):]
    return {'time': row.get('wall',row.get('ts','')), 'category':category, 'task_id':row.get('task_id'), 'func':reply if reply is not None else row.get('request_func'), 'phenomenon':row.get('what',fields.get('what',reason)), 'location':row.get('where',row.get('location',fields.get('where',row.get('operation','')))), 'suggestion':row.get('hint',row.get('suggestion',fields.get('hint',row.get('advice','')))), 'color':color, 'raw':row}

class EventTail:
    def __init__(self, root):
        self.root = Path(root); self.offsets = {}; self.rows = []; self.day = None
    def poll(self, day=None):
        day = day or date.today()
        if day != self.day:
            self.offsets = {}; self.day = day
        folder = self.root / day.isoformat()
        for path in sorted(folder.rglob('events.jsonl')) if folder.exists() else []:
            try:
                old = self.offsets.get(path, 0)
                if path.stat().st_size < old: old = 0
                with path.open('rb') as f:
                    f.seek(old); data = f.read(); end = data.rfind(b'\n')
                    if end < 0: continue
                    self.offsets[path] = old + end + 1
                for line in data[:end+1].splitlines():
                    try:
                        row = json.loads(line)
                        item = classify_event(row) if isinstance(row, dict) else None
                        if item: self.rows.append(item)
                    except (ValueError, UnicodeError): pass
            except OSError: continue
        self.rows.sort(key=lambda x: x['time'], reverse=True)
        self.rows = self.rows[:500]
        return self.rows

def camcal_progress(status):
    return status.get('camcal', {}) if isinstance(status.get('camcal'), dict) else {}

def tooltip_progress(status):
    return status.get('tooltip', {}) if isinstance(status.get('tooltip'), dict) else {}


def preview_controls(status, age_s, main_mode=None):
    preview = status.get('preview') if isinstance(status.get('preview'), dict) else {}
    live = bool(status.get('pid')) and age_s is not None and age_s <= 3
    running = bool(status.get('child_pid')) or status.get('child_state') in ('启动中', '运行', '就绪')
    from nest_clear.preview.eligibility import preview_eligibility
    enabled, reason = preview_eligibility(status, age_s, main_mode, 'main')
    if live and running and main_mode is None and not enabled:
        reason = '主程序运行中，预览不可用'
    on = bool(preview.get('on'))
    mode = preview.get('mode', 'preview')
    return {'enabled': enabled, 'on': on, 'mode': mode, 'preview_enabled': enabled and (not on or mode == 'preview'), 'focus_enabled': enabled and (not on or mode == 'focus'), 'white_balance_enabled': enabled and on and mode == 'focus', 'reason': reason}


def edge_corner_label(corner):
    # session 中 2 是左下，3 是右下；不能按顺时针序号直接索引。
    return {0: '左上', 1: '右上', 2: '左下', 3: '右下'}.get(corner, str(corner))


def tooltip_vector(value, *, signed=False, labels=None, digits=3, compact=False):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return '—'
    try:
        nums = [float(v) for v in value]
        if not all(math.isfinite(v) for v in nums): return '—'
    except (TypeError, ValueError):
        return '—'
    if labels:
        return ' '.join(f'{axis} {format_number(v, digits, signed=signed)}' for axis, v in zip(labels, nums))
    parts = [format_number(v, digits, signed=signed) for v in nums]
    return (' ' if compact else ' / ').join(parts)


def tooltip_deviation(value, *, compact=False):
    if not isinstance(value, dict): return '—'
    parts = []
    for name, label, short in (('front', '主相机', '主'), ('side', '辅相机', '辅')):
        pair = value.get(name)
        try:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2: raise ValueError
            u, v = map(float, pair)
            if not math.isfinite(u) or not math.isfinite(v): raise ValueError
            u_text, v_text = (format_number(n, signed=True) for n in (u, v))
            parts.append(f'{short} {u_text}/{v_text}' if compact else f'{label} 横向 {u_text} 竖向 {v_text} mm')
        except (ValueError, TypeError):
            parts.append(f'{short} —' if compact else f'{label} —')
    return ' '.join(parts) if compact else '；'.join(parts)


def tooltip_pixels(value):
    if not isinstance(value, dict): return '—'
    def pair(name):
        try:
            p = value[name]
            if len(p) != 2: return '—'
            return ','.join(format_number(x, 0) for x in p)
        except (KeyError, TypeError, ValueError): return '—'
    return pair('front') + ' / ' + pair('side')


def tooltip_quality(status, quality):
    status = status if status in ('未测', '本次测得', '已存') else '未测'
    if status == '未测' or not isinstance(quality, dict): return status
    rms = quality.get('rms_px') or {}
    try:
        front, side, condition = float(rms['front']), float(rms['side']), float(quality['condition_number'])
        if not all(math.isfinite(x) for x in (front, side, condition)): raise ValueError
        return f'{status}（RMS 主 {format_number(front, 2)}px 辅 {format_number(side, 2)}px，条件数 {format_number(condition, 1)}）'
    except (TypeError, ValueError, KeyError): return status


def tooltip_time(value):
    dt = parse_time(value)
    return dt.strftime('%H:%M:%S') if dt else '—'


def _with_baseline_detector_warning(state):
    baseline = state.get('baseline') or {}
    existing = baseline.get('existing') or {}
    flagged = [(name, item['detector_warning']) for name, item in existing.items()
               if item.get('detector_warning')]
    if not flagged:
        return state
    notice = '；'.join(f'工具{name[-1]}：{warning}' for name, warning in flagged)
    preview = baseline.get('preview') or {}
    notice_hidden = (baseline.get('phase') in ('idle', 'done', 'prepare')
                     and any((item or {}).get('on') or (item or {}).get('error') for item in preview.values()))
    camera_warning = any(item.get('camera_differences') for item in existing.values())
    field = 'message' if notice_hidden or camera_warning else 'notice'
    value = '　'.join(part for part in (notice, baseline.get(field)) if part)
    return {**state, 'baseline': {**baseline, field: value}}


def supervisor_tooltip_main(main, supervisor, tooltip, *, supervisor_age=None, now=None):
    """Only show preparation state from the live, current supervisor."""
    if supervisor_live(supervisor, supervisor_age) and (supervisor.get('child_pid') or supervisor.get('child_state') in ('启动中', '运行', '就绪')):
        return _with_baseline_detector_warning(main)
    result = dict(main)
    result['mode'] = '空闲'
    pid = supervisor.get('pid')
    alive = False
    if type(pid) is int and pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except (ProcessLookupError, OSError):
            pass
    updated = parse_time(tooltip.get('updated_at'))
    now_dt, _ = _now_and_epoch(now)
    fresh = updated is not None and (now_dt - updated).total_seconds() <= SUPERVISOR_STALE_SECONDS
    if (supervisor_live(supervisor, supervisor_age) and alive and fresh
            and tooltip.get('pid') == pid and tooltip.get('status') == 'running'
            and tooltip.get('start_time') == supervisor.get('start_time')):
        result['baseline'] = tooltip.get('baseline') or {}
        result['command_seq'] = tooltip.get('command_seq', 0)
    else:
        result['baseline'] = {'phase': 'unknown', 'message': '对刀状态未知或已过期', 'preview': {}}
        result['command_seq'] = 0
    return _with_baseline_detector_warning(result)


def depthcal_stage_info(data):
    """格式化下针深度标定阶段徽标文本与配色。"""
    phase = data.get('phase') or 'idle'
    message = data.get('message') or ''
    if phase == 'idle':
        return '阶段：空闲（等待 PLC 160）', '#eef2f6', '#1d2939'
    if phase == 'done':
        return '阶段：完成（已回 161）', '#dcfce7', '#166534'
    if phase == 'failed':
        return '阶段：失败（已回 999）', '#fee2e2', '#b91c1c'
    if phase == 'plane':
        return '阶段：平面判定', '#e0f2fe', '#0369a1'
    if phase in ('center', 'coarse', 'overshoot', 'fine', 'peak'):
        detail = message or {
            'center': '居中',
            'coarse': '粗扫',
            'overshoot': '细扫 · 抬升',
            'fine': '细扫',
            'peak': '峰位采图',
        }[phase]
        return f'阶段：{detail}', '#dbeafe', '#1d4ed8'
    return f'阶段：{phase}', '#eef2f6', '#1d2939'


def depthcal_fine_plan_text(data):
    """格式化当前细扫半宽、步距文本与回退/Z截断提示。"""
    plan = data.get('fine_plan')
    results = data.get('results') or []
    phase = data.get('phase') or 'idle'
    if (not isinstance(plan, dict) or not plan) and phase == 'done':
        for res in reversed(results):
            if isinstance(res.get('fine_plan'), dict) and res['fine_plan']:
                plan = res['fine_plan']
                break
    if isinstance(plan, dict) and plan:
        half = plan.get('half_width_mm')
        step = plan.get('fine_step_mm')
        mode = plan.get('mode') or 'auto'
        coarse_fwhm = plan.get('coarse_fwhm_mm')
        half_txt = f'±{float(half):.3f} mm' if isinstance(half, (int, float)) else '—'
        step_txt = f'{float(step):.3f} mm' if isinstance(step, (int, float)) else '—'
        if mode == 'auto' and isinstance(coarse_fwhm, (int, float)):
            tag = f'auto, 粗扫 FWHM {float(coarse_fwhm):.3f} mm'
        elif mode == 'fixed_fallback':
            tag = '回退 fixed'
        else:
            tag = str(mode)
        summary = f'细扫：{half_txt} / 步距 {step_txt}（{tag}）'
    else:
        summary = '细扫：等待粗扫完成后确定'

    notices = []
    seen = set()
    sources = []
    label = data.get('label')
    if isinstance(plan, dict) and plan:
        sources.append((label or '当前点', plan))
    for res in results:
        rp = res.get('fine_plan')
        if isinstance(rp, dict) and rp:
            sources.append((res.get('label') or '测点', rp))
    for lbl, fp in sources:
        if fp.get('fallback'):
            item = f"{lbl} 粗扫拟合失败或 FWHM 无效，已回退 fixed（{fp.get('fallback_reason') or '默认范围'}）"
            if item not in seen:
                seen.add(item)
                notices.append(item)
        if fp.get('truncated'):
            item = f"{lbl} 细扫范围被 Z 约束截断（{fp.get('truncation_note') or '已贴合可用 Z 行程'}）"
            if item not in seen:
                seen.add(item)
                notices.append(item)
    return summary, '；'.join(notices)


def depthcal_feasibility_text(data):
    """格式化对焦重复性可行性结论栏文本及颜色。"""
    feas = data.get('feasibility') or {}
    if feas and feas.get('max_range') is not None:
        max_range = float(feas['max_range'])
        max_std = feas.get('max_std')
        std_text = f'{float(max_std):.4f}' if isinstance(max_std, (int, float)) else '—'
        threshold = float(feas.get('threshold', 0.02))
        insufficient_fits = int(feas.get('insufficient_fits') or 0)
        total_fits = int(feas.get('total_fits') or 0)
        insufficient_items = feas.get('insufficient_items') or []
        if insufficient_fits > 0:
            items_text = '、'.join(insufficient_items[:6]) + ('…' if len(insufficient_items) > 6 else '')
            scope = f'{insufficient_fits}/{total_fits}' if total_fits else str(insufficient_fits)
            detail = f'（{items_text}）' if items_text else ''
            text = (
                f'可行性结论：拟合点不足 {scope} 次{detail}，不计入满足；'
                f'当前最大极差 {max_range:.4f} mm / 最大标准差 {std_text} mm'
                f'（门限 {threshold:.3f} mm）→ 不满足'
            )
            return text, '#fff1f0', '#b42318'
        met = bool(feas.get('met'))
        verdict = '满足' if met else '不满足'
        text = (
            f'可行性结论：最大极差 {max_range:.4f} mm / 最大标准差 {std_text} mm'
            f'（门限 {threshold:.3f} mm）→ 对焦重复性{verdict}下针深度标定误差预算（总 0.05mm，对焦环节约 0.02mm）'
        )
        return (text, '#ecfdf3', '#168448') if met else (text, '#fff1f0', '#b42318')
    results = data.get('results') or []
    if results:
        ranges = [float(r['range']) for r in results if isinstance(r.get('range'), (int, float))]
        stds = [float(r['std']) for r in results if isinstance(r.get('std'), (int, float))]
        insufficient = sum(
            1 for r in results for f in (r.get('fits') or []) if not f.get('quality_ok', True)
        )
        if ranges:
            std_text = f'{max(stds):.4f}' if stds else '—'
            extra = f'，拟合点不足 {insufficient} 次（不计入满足）' if insufficient else ''
            return (
                f'可行性结论：已完成 {len(results)} 点，当前最大极差 {max(ranges):.4f} mm / 最大标准差 {std_text} mm{extra}'
                '（待全部测点完成后判定，误差预算：总 0.05mm，对焦环节约 0.02mm）',
                '#fff7ed' if insufficient else '#f8fafc',
                '#b45309' if insufficient else '#475467',
            )
    return '可行性结论：等待测点完成后评估（总误差预算 0.05mm，对焦环节约 0.02mm）', '#f8fafc', '#667085'


def depthcal_subplot_layout(labels, width, height):
    """计算多测点清晰度曲线子图的网格列数、行数与像素区域。"""
    count = max(1, len(labels))
    cols = 3 if count > 4 else 2 if count > 1 else 1
    rows = max(1, math.ceil(count / cols))
    top_pad = 24
    pad = 8
    cell_w = max(80, (width - pad * (cols + 1)) / cols)
    cell_h = max(60, (height - top_pad - pad * (rows + 1)) / rows)
    boxes = []
    for idx, label in enumerate(labels):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = top_pad + pad + r * (cell_h + pad)
        boxes.append((label, (x0, y0, x0 + cell_w, y0 + cell_h)))
    return boxes

"""文件和事件到显示数据的转换，不依赖 tkinter。"""
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from nest_clear.display import format_number

BITS = ('手动模式','自动模式','循环启动','循环暂停','循环停止','上电完成','急停中','初始化中','初始化完成','信息提示','一般报警','严重故障')


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
    tray = val('tray_id','tray'); view = val('view_index','view'); total = val('total_views','total')
    if isinstance(view, int) and 'view_index' in p: view += 1
    return f"第 {tray} 盘 · 视野 {view}/{total} · 粗拍 {val('coarse_shots')} 次 · 可去 {val('removable_count', 'n0')} · 红框 {val('red_count')} · 已下发 {val('sent_count', 'attempts')}"

def status_indicators(s, m, now=None):
    plc_text, plc_color = relative_time(s.get('plc_last_read'), now)
    hb_text, hb_color = relative_time(s.get('heartbeat_last_write'), now)
    if s.get('plc_connected') is False: plc_color = 'red'
    if s.get('heartbeat_interrupted'): hb_color = 'red'
    return {'父程序': ('green' if s.get('pid') else 'gray', str(s.get('pid') or '无数据')), '主程序': ('green' if s.get('child_state') in ('运行','就绪') else 'yellow' if s.get('child_state') == '启动中' else 'gray', str(m.get('mode') or s.get('child_state') or '无数据')), 'PLC 连接': (plc_color, plc_text), '心跳': (hb_color, hb_text), '40932 就绪': ('green' if s.get('algorithm_ready') == 1 else 'gray', str(s.get('algorithm_ready') if s.get('algorithm_ready') is not None else '无数据'))}

EVENTS = {'error','pickacc_failed','tooloffset_failed','tooloffset_quality_warning','camcal_failed','tooltip_failed','supervisor_reply_999','request_abandoned','plc_disconnected','plc_restored','heartbeat_failed','heartbeat_restored','image_dropped','warning'}
def classify_event(row):
    event = str(row.get('event','')).lower()
    text = ' '.join(str(row.get(k,'')) for k in ('reason_code','code','reason','what','operation','event')).lower()
    reply = row.get('reply_func', row.get('func_code'))
    if reply is None and event == 'reply': reply = row.get('func')
    if reply is None and event == 'supervisor_reply_999': reply = 999
    if event not in EVENTS and reply not in (999,131,82,92) and not any(x in text for x in ('readback','回读不一致','采图失败','capture_failed')):
        return None
    category = '父程序' if row.get('process') == 'supervisor' else '标定' if any(x in text for x in ('camcal','tooltip','tooloffset','pickacc','calib')) or reply in (82,92,131,152) else '通信' if any(x in text for x in ('plc','heartbeat','readback','回读')) else '相机' if any(x in text for x in ('camera','capture','image','采图')) else '配置' if 'config' in text else '算法'
    color = 'green' if 'restored' in event or '恢复' in event else 'yellow' if event in ('warning','tooloffset_quality_warning','request_abandoned','image_dropped') else 'red'
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


def preview_controls(status, age_s):
    preview = status.get('preview') if isinstance(status.get('preview'), dict) else {}
    live = bool(status.get('pid')) and age_s is not None and age_s <= 3
    running = bool(status.get('child_pid')) or status.get('child_state') in ('启动中', '运行', '就绪')
    reason = '父程序未运行' if not live else '挑毛主流程运行中，预览不可用' if running else None
    on = bool(preview.get('on'))
    mode = preview.get('mode', 'preview')
    enabled = live and not running
    return {'enabled': enabled, 'on': on, 'mode': mode, 'preview_enabled': enabled and (not on or mode == 'preview'), 'focus_enabled': enabled and (not on or mode == 'focus'), 'reason': reason}


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

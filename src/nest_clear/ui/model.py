"""文件和事件到显示数据的转换，不依赖 tkinter。"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

BITS = ('手动模式','自动模式','循环启动','循环暂停','循环停止','上电完成','急停中','初始化中','初始化完成','信息提示','一般报警','严重故障')

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

EVENTS = {'error','tooloffset_failed','tooloffset_quality_warning','camcal_failed','tooltip_failed','supervisor_reply_999','request_abandoned','plc_disconnected','plc_restored','heartbeat_failed','heartbeat_restored','image_dropped','warning'}
def classify_event(row):
    event = str(row.get('event','')).lower()
    text = ' '.join(str(row.get(k,'')) for k in ('reason_code','code','reason','what','operation','event')).lower()
    reply = row.get('reply_func', row.get('func_code'))
    if reply is None and event == 'reply': reply = row.get('func')
    if reply is None and event == 'supervisor_reply_999': reply = 999
    if event not in EVENTS and reply not in (999,131,82,92) and not any(x in text for x in ('readback','回读不一致','采图失败','capture_failed')):
        return None
    category = '父程序' if row.get('process') == 'supervisor' else '标定' if any(x in text for x in ('camcal','tooltip','tooloffset','calib')) or reply in (82,92,131) else '通信' if any(x in text for x in ('plc','heartbeat','readback','回读')) else '相机' if any(x in text for x in ('camera','capture','image','采图')) else '配置' if 'config' in text else '算法'
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
    return {'enabled': live and not running, 'on': bool(preview.get('on')), 'reason': reason}


def edge_corner_label(corner):
    # session 中 2 是左下，3 是右下；不能按顺时针序号直接索引。
    return {0: '左上', 1: '右上', 2: '左下', 3: '右下'}.get(corner, str(corner))

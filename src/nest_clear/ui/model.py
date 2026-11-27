"""文件和事件到显示数据的转换，不依赖 tkinter。"""
import json
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from nest_clear.display import format_number
from nest_clear.paths import tooltip_preview_file
from nest_clear.baseline.focus_rules import (
    BASELINE_RULE,
    CAMERA_AXES,
    EDGE_MARGIN_MESSAGE,
    GEOMETRY_SUMMARY,
    PEAK_OVERSHOOT_HINT,
    STAGE_ASCENDING_TEXT,
    STAGE_OVERSHOT_TEXT,
    STAGE_READY_TEXT,
    evaluate_baseline_readiness,
    evaluate_camera_readiness,
)

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


def baseline_focus_status(live, *, good_px=2.0, warn_px=3.0, axis_rms_limit=0.7):
    """根据边缘宽度判断对焦状态，并格式化对焦状态条摘要（不依赖 cv2/numpy/yaml）。"""
    try:
        good = float(good_px)
        if not math.isfinite(good) or good <= 0:
            good = 2.0
    except (TypeError, ValueError):
        good = 2.0
    try:
        warn = float(warn_px)
        if not math.isfinite(warn) or warn < good:
            warn = max(3.0, good)
    except (TypeError, ValueError):
        warn = max(3.0, good)
    src = live if isinstance(live, dict) else {}
    det_raw = src.get('detection_status')
    det_txt = '未检出' if det_raw in (None, '', '未识别', '未检出') else str(det_raw)
    edge_px = src.get('edge_width_px')
    edge_um = src.get('edge_width_um')
    contrast = src.get('contrast')
    axis_rms = src.get('axis_rms_px')
    lim = src.get('axis_rms_limit')
    lim_val = float(lim) if isinstance(lim, (int, float)) and math.isfinite(lim) and lim > 0 else axis_rms_limit
    valid_edge = (
        bool(src)
        and det_txt != '未检出'
        and isinstance(edge_px, (int, float))
        and not isinstance(edge_px, bool)
        and math.isfinite(edge_px)
        and edge_px > 0
    )
    if not valid_edge:
        level, badge, color_key = 'none', '未检出', 'gray'
    elif edge_px <= good:
        level, badge, color_key = 'good', '✔ 已对焦', 'green'
    elif edge_px <= warn:
        level, badge, color_key = 'warn', '⚠ 稍虚', 'yellow'
    else:
        level, badge, color_key = 'bad', '✖ 未对焦', 'red'
    pct = src.get('percent')
    pct_txt = f'{float(pct):.0f}%' if isinstance(pct, (int, float)) and not isinstance(pct, bool) and math.isfinite(pct) else '—%'
    edge_px_txt = f'{float(edge_px):.2f} px' if isinstance(edge_px, (int, float)) and not isinstance(edge_px, bool) and math.isfinite(edge_px) else '— px'
    edge_um_txt = f'{float(edge_um):.1f} µm' if isinstance(edge_um, (int, float)) and not isinstance(edge_um, bool) and math.isfinite(edge_um) else '— µm'
    contrast_txt = f'{float(contrast):.0f}' if isinstance(contrast, (int, float)) and not isinstance(contrast, bool) and math.isfinite(contrast) else '—'
    rms_txt = f'{float(axis_rms):.2f} px' if isinstance(axis_rms, (int, float)) and not isinstance(axis_rms, bool) and math.isfinite(axis_rms) else '— px'
    lim_txt = f'{float(lim_val):g}' if isinstance(lim_val, (int, float)) and math.isfinite(lim_val) else '0.7'
    summary = f'清晰度 {pct_txt}（相对本次峰值）· 边缘 {edge_px_txt}（{edge_um_txt}）· 对比度 {contrast_txt} · 轴残差 {rms_txt}（门限 {lim_txt}）· {det_txt}'
    return {
        'level': level,
        'badge': badge,
        'color_key': color_key,
        'summary': summary,
        'detection_label': det_txt,
        'axis_rms_limit': lim_val,
    }


def baseline_preview_display(item, live=None, *, allow_inline=False, session_live=False):
    """Format one preparation or baseline-session view, discarding stale metrics after an error or expiration."""
    error = str(item.get('error') or '')
    if error:
        hint = ('两台相机不要接在同一个 USB 集线器上，换一个主板 USB 口后重开预览。'
                if 'USB' in error or '带宽' in error else '检查相机配置、连接和权限后重开预览。')
        return {'failed': True, 'on': bool(item.get('on')), 'session_live': bool(session_live), 'no_live_data': False,
                'button': '预览失败，点击重试', 'message': error, 'hint': hint,
                'metrics': {'percent': '—', 'current': '—', 'peak': '—', 'edge_um': '—',
                            'edge_px': '—', 'contrast': '—', 'source': '—', 'exposure': '—'},
                'history': []}
    on = bool(item.get('on'))
    if not on and not session_live:
        return {'failed': False, 'on': False, 'session_live': False, 'no_live_data': False, 'message': '预览关闭',
                'metrics': None, 'history': []}
    src = live if isinstance(live, dict) and live else (item if on and allow_inline and 'percent' in item else None)
    if not src:
        exposure_error = item.get('exposure_error')
        message = str(exposure_error) if exposure_error else ('等待实时检测数据' if session_live and not on else '无实时数据')
        return {'failed': False, 'on': on, 'session_live': bool(session_live), 'no_live_data': True, 'message': message,
                'metrics': {'percent': '—', 'current': '—', 'peak': '—', 'edge_um': '—',
                            'edge_px': '—', 'contrast': '—', 'source': '—', 'exposure': '—'},
                'history': []}
    return {'failed': False, 'on': on, 'session_live': bool(session_live), 'no_live_data': False, 'live': src,
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


def depthcal_session_key(status):
    """提取下针深度标定会话标识，用于识别新会话（含两次轮询间已结束的会话）。"""
    if not isinstance(status, dict):
        return None
    depth = status.get('depthcal')
    if not isinstance(depth, dict) or not depth:
        return None
    session = depth.get('session')
    if session:
        return ('session', str(session))
    phase = depth.get('phase')
    if phase and phase != 'idle':
        token = depth.get('task_id') or depth.get('seq_event') or depth.get('wall') or depth.get('ts')
        if token is not None:
            return ('event', str(token))
        return ('phase', str(phase), str(depth.get('reason_code') or ''), str(depth.get('what') or ''))
    return None


def next_auto_page(status, last_mode, last_depth_session):
    """根据 mode 变化与新下针深度标定会话决定自动切页目标，返回 (page, new_mode, new_depth_session)。"""
    mode = status.get('mode') if isinstance(status, dict) else None
    depth_session = depthcal_session_key(status)
    mode_page = page_for_mode(status) if mode != last_mode else None
    new_depth = depth_session is not None and depth_session != last_depth_session
    target_page = mode_page
    if new_depth and page_for_mode(status) in (None, '下针深度标定'):
        target_page = '下针深度标定'
    return target_page, mode, depth_session


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


def main_running(supervisor, main=None, *, supervisor_age=None, main_alive_age=None):
    """根据父程序与主程序状态判断主程序当前是否在运行。"""
    if isinstance(supervisor, dict) and supervisor:
        if not supervisor_live(supervisor, supervisor_age):
            return False
        return bool(supervisor.get('child_pid')) or supervisor.get('child_state') in ('启动中', '运行', '就绪')
    if main_alive_age is not None and main_alive_age <= SUPERVISOR_STALE_SECONDS:
        return bool(isinstance(main, dict) and main.get('pid')) or True
    return False


def command_timeout_message(name, supervisor, main=None, *, supervisor_age=None, main_alive_age=None):
    """命令 5 秒未确认时按主程序运行状态给出准确提示。"""
    if main_running(supervisor, main, supervisor_age=supervisor_age, main_alive_age=main_alive_age):
        return f'主程序未响应命令【{name}】（已放弃，可重试），请查看日志'
    return '主程序未运行，请先让 PLC 初始化'


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

EVENTS = {'arrival_warning','depthcal_arrival_warning','depthcal_exposure_warning','tooltip_settings_warning','tooltip_preview_error','tooltip_exposure_failed','error','pickacc_failed','tooloffset_failed','tooloffset_quality_warning','camcal_failed','tooltip_failed','supervisor_reply_999','request_abandoned','plc_disconnected','plc_restored','heartbeat_failed','heartbeat_restored','image_dropped','warning'}
def classify_event(row):
    event = str(row.get('event','')).lower()
    text = ' '.join(str(row.get(k,'')) for k in ('reason_code','code','reason','what','operation','event')).lower()
    reply = row.get('reply_func', row.get('func_code'))
    if reply is None and event == 'reply': reply = row.get('func')
    if reply is None and event == 'supervisor_reply_999': reply = 999
    if event not in EVENTS and reply not in (999, 131, 99, 82) and not any(x in text for x in ('readback','回读不一致','采图失败','capture_failed')):
        return None
    category = '父程序' if row.get('process') == 'supervisor' else '标定' if any(x in text for x in ('camcal','tooltip','tooloffset','pickacc','calib','depthcal')) or reply in (82, 99, 131, 152) else '通信' if any(x in text for x in ('plc','heartbeat','readback','回读')) else '相机' if any(x in text for x in ('camera','capture','image','采图')) else '配置' if 'config' in text else '算法'
    color = 'green' if 'restored' in event or '恢复' in event else 'yellow' if event in ('arrival_warning','depthcal_arrival_warning','depthcal_exposure_warning','warning','tooltip_settings_warning','tooloffset_quality_warning','request_abandoned','image_dropped') else 'red'
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
    summary = data.get('arrival_summary')
    has_summary = isinstance(summary, dict) and summary.get('total_anomalies') is not None
    anom = int(summary.get('total_anomalies') or 0) if has_summary else 0
    total_arr = int(summary.get('total_arrivals') or 0) if has_summary else 0
    if phase == 'idle':
        return '阶段：空闲（等待 PLC 160）', '#eef2f6', '#1d2939'
    if phase == 'done':
        return '阶段：完成（已回 161）', '#dcfce7', '#166534'
    if phase == 'failed':
        suffix = f' · 到位异常 {anom}/{total_arr} 次' if (has_summary and total_arr > 0) else ''
        return f'阶段：失败（已回 999）{suffix}', '#fee2e2', '#b91c1c'
    if phase == 'plane':
        suffix = f' · 到位异常 {anom} 次' if has_summary else ''
        return f'阶段：平面判定{suffix}', '#e0f2fe', '#0369a1'
    if phase in ('center', 'exposure', 'coarse_up', 'coarse_overshoot', 'coarse', 'overshoot', 'fine', 'peak'):
        detail = message or {
            'center': '居中',
            'exposure': '测点自动曝光',
            'coarse_up': '粗找 · 上行',
            'coarse_overshoot': '粗扫 · 抬升',
            'coarse': '粗扫 · 下行',
            'overshoot': '细扫 · 抬升',
            'fine': '细扫',
            'peak': '峰位采图',
        }[phase]
        suffix = f' · 到位异常 {anom} 次' if has_summary else ''
        return f'阶段：{detail}{suffix}', '#dbeafe', '#1d4ed8'
    return f'阶段：{phase}', '#eef2f6', '#1d2939'


def depthcal_exposure_display(data):
    """格式化当前测点锁定曝光与 ROI 均值/p99 实时显示文本。"""
    sample = data.get('sample') if isinstance(data.get('sample'), dict) else {}
    results = data.get('results') or []
    last_res = results[-1] if results and isinstance(results[-1], dict) else {}
    cur_exp = data.get('current_exposure_us')
    if cur_exp is None:
        cur_exp = sample.get('exposure_us', last_res.get('exposure_us'))
    roi_mean = data.get('roi_mean')
    if roi_mean is None:
        roi_mean = sample.get('roi_mean', last_res.get('roi_mean'))
    roi_p99 = data.get('roi_p99')
    if roi_p99 is None:
        roi_p99 = sample.get('roi_p99', last_res.get('roi_p99'))
    mode = data.get('point_exposure_mode') or data.get('exposure_mode') or 'auto'
    mode_tag = '自动' if mode == 'auto' else '手动'
    exp_txt = f'{float(cur_exp):.0f} µs（{mode_tag}）' if isinstance(cur_exp, (int, float)) else '— µs'
    mean_txt = f'{float(roi_mean):.1f}' if isinstance(roi_mean, (int, float)) else '—'
    p99_txt = f'{float(roi_p99):.1f}' if isinstance(roi_p99, (int, float)) else '—'
    return f'当前曝光：{exp_txt}  |  ROI 均值 {mean_txt} / p99 {p99_txt}'


def depthcal_point_arrival_text(item):
    """格式化单点结果的“到位超差”列文本及是否异常标记。"""
    arrival = item.get('arrival') if isinstance(item, dict) else None
    if not isinstance(arrival, dict):
        return '—', False
    anomalies = int(arrival.get('anomalies') or 0)
    total = int(arrival.get('total') or 0)
    if anomalies <= 0:
        return '—', False
    max_delta = arrival.get('max_delta_mm')
    max_axis = arrival.get('max_delta_axis')
    if isinstance(max_delta, (int, float)) and max_axis:
        detail = f'最大 {max_axis} {float(max_delta):+.3f}'
    else:
        detail = '未稳定'
    return f'{anomalies}/{total} 次 · {detail}', True


def depthcal_arrival_summary_text(data):
    """格式化下针深度标定表上方的到位汇总文本与配色。"""
    summary = data.get('arrival_summary')
    results = data.get('results') or []
    phase = data.get('phase') or 'idle'
    if not isinstance(summary, dict) or not summary.get('total_arrivals'):
        pts_arr = [(r.get('label'), r.get('arrival')) for r in results if isinstance(r.get('arrival'), dict)]
        if not pts_arr:
            return '到位：等待测点到位数据（详见 arrivals.csv）', '#f8fafc', '#667085'
        total_arr = sum(int(a.get('total') or 0) for _, a in pts_arr)
        total_anom = sum(int(a.get('anomalies') or 0) for _, a in pts_arr)
        anom_pts = [lbl for lbl, a in pts_arr if int(a.get('anomalies') or 0) > 0 and lbl]
        with_d = [a for _, a in pts_arr if isinstance(a.get('max_abs_delta_mm'), (int, float))]
        worst = max(with_d, key=lambda a: a['max_abs_delta_mm']) if with_d else None
        el_vals = [float(a['avg_elapsed_s']) for _, a in pts_arr if isinstance(a.get('avg_elapsed_s'), (int, float))]
        summary = {
            'total_arrivals': total_arr,
            'total_anomalies': total_anom,
            'anomaly_points': anom_pts,
            'max_delta_mm': worst.get('max_delta_mm') if worst else None,
            'max_delta_axis': worst.get('max_delta_axis') if worst else None,
            'position_tolerance_mm': 0.01,
            'avg_elapsed_s': (sum(el_vals) / len(el_vals)) if el_vals else None,
        }

    anomalies = int(summary.get('total_anomalies') or 0)
    tol = float(summary.get('position_tolerance_mm') or 0.01)
    avg_el = summary.get('avg_elapsed_s')
    avg_txt = f'{float(avg_el):.2f}' if isinstance(avg_el, (int, float)) else '0.00'
    if anomalies > 0:
        pts = summary.get('anomaly_points') or []
        pts_txt = '、'.join(pts) if pts else '当前测点'
        max_delta = summary.get('max_delta_mm')
        max_axis = summary.get('max_delta_axis')
        if isinstance(max_delta, (int, float)) and max_axis:
            detail = f'，最大 {max_axis} {float(max_delta):+.3f} mm'
        else:
            detail = '，存在未稳定'
        return f'到位超差点：{pts_txt}（共 {anomalies} 次{detail}，详见 arrivals.csv）', '#fef3c7', '#916300'
    text = f'到位：全部在 ±{tol:.2f} mm 内（平均等待 {avg_txt} s）'
    if phase == 'done':
        return text, '#ecfdf3', '#166534'
    return text, '#f8fafc', '#475467'


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
        step_txt = f'{float(step):.3f} mm' if isinstance(step, (int, float)) else '—'
        if isinstance(plan.get('blur_low_z'), (int, float)) and isinstance(plan.get('blur_high_z'), (int, float)):
            lo_z = float(plan['blur_low_z'])
            hi_z = float(plan['blur_high_z'])
            fwhm_tag = f', 粗扫 FWHM {float(coarse_fwhm):.3f} mm' if isinstance(coarse_fwhm, (int, float)) else ''
            summary = f'细扫：[{lo_z:.2f}, {hi_z:.2f}] mm / 步距 {step_txt}（{mode}{fwhm_tag}）'
        else:
            half_txt = f'±{float(half):.3f} mm' if isinstance(half, (int, float)) else '—'
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


def depthcal_surface_text(data):
    """格式化下针深度标定测量面类型显示文本。"""
    label = data.get('target_type_label')
    if label:
        return f'测量面：{label}'
    target_type = data.get('target_type')
    if target_type == 'film':
        return '测量面：菲林'
    if target_type == 'charuco':
        thickness = data.get('board_thickness_mm')
        if isinstance(thickness, (int, float)):
            return f'测量面：ChArUco 标定板（厚 {float(thickness):.2f}mm）'
        return '测量面：ChArUco 标定板'
    if (data.get('phase') or 'idle') == 'idle':
        return '测量面：自动检测（菲林 / ChArUco）'
    return '测量面：自动检测（识别中）'


def depthcal_board_fit_text(data):
    """格式化 ChArUco 板二维位姿拟合残差与安全区文本；非 ChArUco 模式返回空串。"""
    fit = data.get('board_fit')
    if not isinstance(fit, dict) or not fit:
        return ''
    corners = fit.get('corner_count')
    angle = fit.get('angle_deg')
    rms = fit.get('rms_residual_mm')
    mx = fit.get('max_residual_mm')
    parts = []
    if corners is not None:
        parts.append(f'{int(corners)} 角点')
    if isinstance(angle, (int, float)):
        parts.append(f'转角 {float(angle):+.2f}°')
    if isinstance(rms, (int, float)):
        parts.append(f'RMS {float(rms):.4f} mm')
    if isinstance(mx, (int, float)):
        parts.append(f'最大 {float(mx):.4f} mm')
    if not parts:
        return ''
    keepout = fit.get('board_keepout_mm', data.get('board_keepout_mm', 10.0))
    keepout_str = f'{float(keepout):g}' if isinstance(keepout, (int, float)) else '10'
    return f"板位姿拟合：{' · '.join(parts)} · 安全区：板内缩 {keepout_str}mm"


def depthcal_point_position(label, data=None, item=None):
    """读取测点对应的板面方位名（如 左上/中心/上中）。"""
    if isinstance(item, dict) and item.get('position_name'):
        return str(item['position_name'])
    positions = (data or {}).get('point_positions') if isinstance(data, dict) else None
    if isinstance(positions, dict) and label in positions and positions[label]:
        return str(positions[label])
    return ''


def depthcal_feasibility_text(data):
    """格式化对焦重复性可行性结论栏文本及颜色（ChArUco 模式附带板面平面残差判级）。"""
    feas = data.get('feasibility') or {}
    target_type = data.get('target_type') or feas.get('target_type')
    plane = data.get('plane') or {}
    if feas and feas.get('max_range') is not None:
        max_range = float(feas['max_range'])
        max_std = feas.get('max_std')
        std_text = f'{float(max_std):.4f}' if isinstance(max_std, (int, float)) else '—'
        threshold = float(feas.get('threshold', 0.02))
        insufficient_fits = int(feas.get('insufficient_fits') or 0)
        total_fits = int(feas.get('total_fits') or 0)
        insufficient_items = feas.get('insufficient_items') or []
        if target_type == 'charuco':
            plane_res = plane.get('max_residual')
            if not isinstance(plane_res, (int, float)):
                plane_res = feas.get('plane_max_residual')
            if isinstance(plane_res, (int, float)):
                res_val = float(plane_res)
                res_str = f'{res_val:.3f}'
                plane_verdict = '可行' if res_val <= 0.02 else ('勉强' if res_val <= 0.05 else '不可行')
            else:
                res_val = None
                res_str = '—'
                plane_verdict = feas.get('plane_verdict') or '待定'
            if insufficient_fits > 0:
                items_text = '、'.join(insufficient_items[:6]) + ('…' if len(insufficient_items) > 6 else '')
                scope = f'{insufficient_fits}/{total_fits}' if total_fits else str(insufficient_fits)
                detail = f'（{items_text}）' if items_text else ''
                rep_part = (
                    f'各点重复性：拟合点不足 {scope} 次{detail}，'
                    f'最大极差 {max_range:.4f} mm / 最大标准差 {std_text} mm（门限 {threshold:.3f} mm → 不满足）'
                )
            else:
                met = bool(feas.get('met'))
                verdict = '满足' if met else '不满足'
                rep_part = (
                    f'各点重复性：最大极差 {max_range:.4f} mm / 最大标准差 {std_text} mm'
                    f'（门限 {threshold:.3f} mm → {verdict}）'
                )
            text = (
                f'可行性结论：标定板平面残差 {res_str} mm（≤0.02 可行 / ≤0.05 勉强 / >0.05 不可行，当前{plane_verdict}）'
                f' + {rep_part}'
            )
            if insufficient_fits > 0 or not bool(feas.get('met')) or (res_val is not None and res_val > 0.05):
                return text, '#fff1f0', '#b42318'
            if res_val is not None and res_val > 0.02:
                return text, '#fff7ed', '#b45309'
            return text, '#ecfdf3', '#168448'

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
            if target_type == 'charuco':
                return (
                    f'可行性结论：已完成 {len(results)} 点，当前最大极差 {max(ranges):.4f} mm / 最大标准差 {std_text} mm{extra}'
                    '（待全部测点完成后按标定板平面残差 ≤0.02 可行 / ≤0.05 勉强 / >0.05 不可行 + 各点重复性评估）',
                    '#fff7ed' if insufficient else '#f8fafc',
                    '#b45309' if insufficient else '#475467',
                )
            return (
                f'可行性结论：已完成 {len(results)} 点，当前最大极差 {max(ranges):.4f} mm / 最大标准差 {std_text} mm{extra}'
                '（待全部测点完成后判定，误差预算：总 0.05mm，对焦环节约 0.02mm）',
                '#fff7ed' if insufficient else '#f8fafc',
                '#b45309' if insufficient else '#475467',
            )
    if target_type == 'charuco':
        return '可行性结论：等待测点完成后按标定板平面残差（≤0.02 可行 / ≤0.05 勉强 / >0.05 不可行）+ 各点重复性评估', '#f8fafc', '#667085'
    return '可行性结论：等待测点完成后评估（总误差预算 0.05mm，对焦环节约 0.02mm）', '#f8fafc', '#667085'


def depthcal_curve_dof(points, ratio=0.8):
    """纯 Python 计算清晰度曲线上 >= ratio×峰值 的线性插值景深与交点。"""
    pts = sorted((float(p[0]), float(p[1])) for p in (points or []) if len(p) >= 2)
    if len(pts) < 3:
        return None
    index = max(range(len(pts)), key=lambda i: pts[i][1])
    peak_val = pts[index][1]
    if peak_val <= 0 or not math.isfinite(peak_val):
        return None
    threshold = peak_val * float(ratio)
    crossings = []
    for direction in (-1, 1):
        inner = index
        outer = inner + direction
        while 0 <= outer < len(pts) and pts[outer][1] > threshold:
            inner, outer = outer, outer + direction
        if not 0 <= outer < len(pts) or pts[inner][1] == pts[outer][1]:
            crossings.append(None)
        else:
            frac = (threshold - pts[inner][1]) / (pts[outer][1] - pts[inner][1])
            crossings.append(pts[inner][0] + frac * (pts[outer][0] - pts[inner][0]))
    z_lo, z_hi = crossings
    covered = z_lo is not None and z_hi is not None
    return {
        'dof_mm': float(z_hi - z_lo) if covered else None,
        'dof_ratio': float(ratio),
        'dof_z_lower': z_lo,
        'dof_z_upper': z_hi,
        'dof_covered': covered,
        'threshold': threshold,
        'peak_val': peak_val,
    }


def depthcal_skip_summary_text(data):
    """格式化下针深度标定失败点跳过/补测汇总文本与配色。"""
    summary = data.get('skip_summary')
    failed_points = data.get('failed_points') or []
    results = data.get('results') or []
    if isinstance(summary, dict) and summary.get('text') and (summary.get('failed_count') or summary.get('items')):
        skipped_cnt = int(summary.get('skipped_count') or 0)
        if skipped_cnt > 0:
            return f"跳过/补测：{summary['text'].replace('跳过：', '')}" if summary['text'].startswith('跳过：') else summary['text'], '#fee2e2', '#b91c1c'
        return summary['text'], '#fff7ed', '#b45309'
    if failed_points:
        retest_map = {r.get('retest_of'): r.get('label') for r in results if r.get('retest_of')}
        parts = []
        has_skipped = False
        for fp in failed_points:
            if fp.get('retest_of'):
                continue
            lbl = fp.get('label') or '—'
            reason = fp.get('reason') or fp.get('error_code') or '失败'
            retest_lbl = fp.get('retest_label') or retest_map.get(lbl)
            retest_status = fp.get('retest_status')
            if retest_lbl and (retest_status == 'success' or lbl in retest_map):
                parts.append(f'{lbl}({reason})→补测{retest_lbl}成功')
            elif retest_lbl and retest_status == 'failed':
                has_skipped = True
                parts.append(f'{lbl}({reason})→补测{retest_lbl}失败→跳过')
            else:
                has_skipped = True
                parts.append(f'{lbl}({reason})→跳过')
        text = '跳过/补测：' + '；'.join(parts)
        return (text, '#fee2e2', '#b91c1c') if has_skipped else (text, '#fff7ed', '#b45309')
    return '跳过/补测：无失败点', '#f8fafc', '#475467'


def depthcal_latest_dof_text(data):
    """格式化阶段栏旁的当前测点最新景深 DOF（及 FWHM）显示文本。"""
    label = data.get('label')
    results = data.get('results') or []
    current_fits = [f for f in (data.get('current_fits') or []) if not f.get('failed')]
    matched = next((r for r in results if r.get('label') == label), None) if label else None

    src = None
    tag = ''
    fwhm = None
    if current_fits:
        src = current_fits[-1]
        tag = f"细扫{src.get('repeat', len(current_fits))}"
        fwhm = src.get('fwhm')
    elif matched and matched.get('fits'):
        ok_fits = [f for f in matched['fits'] if not f.get('failed')]
        if ok_fits:
            src = ok_fits[-1]
            tag = f"细扫{src.get('repeat', len(ok_fits))}"
            fwhm = src.get('fwhm')
    else:
        coarse_fit = data.get('coarse_fit') or (matched.get('coarse_fit') if matched else None)
        coarse_dof = data.get('coarse_dof') or (matched.get('coarse_dof') if matched else None)
        if isinstance(coarse_fit, dict) and coarse_fit:
            src = coarse_fit
            tag = '粗扫估计'
            fwhm = coarse_fit.get('fwhm')
        elif isinstance(coarse_dof, dict) and coarse_dof:
            src = coarse_dof
            tag = '粗扫估计'
        else:
            curves = data.get('curves') or {}
            coarse_pts = (curves.get(label) or {}).get('coarse') if label else None
            if coarse_pts:
                est = depthcal_curve_dof(coarse_pts, data.get('dof_ratio') or 0.8)
                if est and est.get('dof_covered'):
                    src = est
                    tag = '粗扫估计'

    if not isinstance(src, dict):
        return '景深：—（FWHM —）'
    dof_mm = src.get('dof_mm')
    covered = src.get('dof_covered')
    dof_txt = f'{float(dof_mm):.3f} mm' if (covered is not False and isinstance(dof_mm, (int, float))) else '未覆盖'
    fwhm_txt = f'{float(fwhm):.3f} mm' if isinstance(fwhm, (int, float)) else '—'
    return f'景深：{dof_txt}（{tag}，FWHM {fwhm_txt}）'


def depthcal_dof_summary_text(data):
    """格式化全盘景深汇总文本与配色（含 dof_ratio 与未覆盖注明）。"""
    ratio = float(data.get('dof_ratio') or 0.8)
    summary = data.get('dof_summary')
    results = data.get('results') or []
    if isinstance(summary, dict) and summary.get('total_points') is not None:
        uncov = int(summary.get('uncovered_fits') or 0)
        uncov_items = summary.get('uncovered_items') or []
        uncov_note = ''
        if uncov > 0:
            items_txt = '、'.join(uncov_items[:6]) + ('…' if len(uncov_items) > 6 else '')
            detail = f'（{items_txt}）' if items_txt else ''
            uncov_note = f'；未覆盖 {uncov}/{summary.get("total_fits", 0)} 次{detail}，不计入统计'
        if summary.get('mean_mm') is not None:
            text = (
                f"全盘景深汇总（dof_ratio={ratio:.2f}）：均值 {summary['mean_mm']:.3f} mm / "
                f"最小 {summary['min_mm']:.3f} mm / 最大 {summary['max_mm']:.3f} mm"
                f"（有效 {summary['covered_points']}/{summary['total_points']} 点{uncov_note}）"
            )
            return (text, '#fff7ed', '#b45309') if uncov > 0 else (text, '#ecfdf3', '#166534')
        text = (
            f"全盘景深汇总（dof_ratio={ratio:.2f}）：未覆盖"
            f"（0/{summary['total_points']} 点有效{uncov_note}）"
        )
        return text, '#fff7ed', '#b45309'

    if results:
        dofs = [float(r['dof_mm']) for r in results if isinstance(r.get('dof_mm'), (int, float))]
        uncov = sum(int(r.get('dof_uncovered') or 0) for r in results)
        if dofs:
            extra = f'，未覆盖 {uncov} 次（不计入）' if uncov else ''
            return (
                f'全盘景深（dof_ratio={ratio:.2f}）：已完成 {len(results)} 点，'
                f'均值 {sum(dofs)/len(dofs):.3f} / 最小 {min(dofs):.3f} / 最大 {max(dofs):.3f} mm{extra}',
                '#fff7ed' if uncov else '#f8fafc',
                '#b45309' if uncov else '#475467',
            )
        return (
            f'全盘景深（dof_ratio={ratio:.2f}）：已完成 {len(results)} 点，均未覆盖（不计入统计）',
            '#fff7ed',
            '#b45309',
        )
    return f'全盘景深（dof_ratio={ratio:.2f}）：等待测点完成后汇总', '#f8fafc', '#667085'


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

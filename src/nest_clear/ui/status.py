"""由事件生成主程序状态；写入失败不传播到业务流程。"""
import json
import logging
import os
import threading
import time
from nest_clear.paths import main_status_file

log = logging.getLogger(__name__)
class MainStatus:
    def __init__(self, root):
        self.path = main_status_file(root); self.lock = threading.Lock(); self.last = 0; self.timer = None
        self.data = {'pid':os.getpid(), 'mode':'空闲', 'task_id':None, 'request_func':None, 'last_reply_func':None, 'last_reply_time':None, 'recent_replies':[], 'picking':{}, 'camcal':{}, 'tooltip':{}, 'tooloffset':{}, 'pickacc':{}, 'depthcal':{}, 'baseline':{}, 'command_seq':0}
        self.write()
    def write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.data, ensure_ascii=False, default=str), encoding='utf-8')
            os.replace(temp, self.path)
        except Exception: log.exception('写主程序状态失败')
    def flush(self):
        with self.lock:
            self.write(); self.last = time.monotonic(); self.timer = None
    def observe(self, row):
        with self.lock:
            e = row.get('event',''); d = self.data
            if e == 'film_mode_tray': d['film_mode'] = bool(row.get('enabled'))
            if e == 'mode_change': d['mode'] = row.get('mode','空闲')
            if e == 'request':
                d.update(task_id=row.get('task_id'), request_func=row.get('func'), plc_tool_id=row.get('plc_tool_id'))
                if row.get('func') == 80 and d['camcal'].get('event') not in ('camcal_start', 'camcal_goto_start', 'camcal_step', 'camcal_glare_warning'):
                    d['camcal'] = {'event': 'camcal_starting'}
            if e == 'reply':
                reply = row.get('reply_func', row.get('func'))
                if reply is not None:
                    if d['camcal'].get('event') == 'camcal_starting':
                        d['camcal']['event'] = 'camcal_aborted'
                    d.update(last_reply_func=reply, last_reply_time=row.get('wall'), last_reply_task_id=row.get('task_id'), last_reply_request_func=row.get('request_func', d.get('request_func')))
                    d['recent_replies'].insert(0, {'time': row.get('wall'), 'task_id': row.get('task_id'),
                        'request_func': row.get('request_func', d.get('request_func')), 'reply_func': reply})
                    del d['recent_replies'][5:]
            if e in ('work_cycle','detection','capture','scan_plan','tray_end'): d['picking'].update({k:v for k,v in row.items() if k not in ('event','pid','mono')})
            if e == 'camcal_start': d['camcal'] = {}
            if e == 'camcal_glare_warning':
                d['camcal']['glare_warning'] = {k: row[k] for k in ('image_index', 'bboxes', 'excluded_corners') if k in row}
            elif e.startswith('camcal_'):
                d['camcal'].update(row)
                if e == 'camcal_step':
                    d['camcal'].pop('glare_warning', None)
                    count = max(0, int(row.get('sequence', 1)) - 1)
                    d['camcal']['completed_indices'] = list(range(min(count, len(row.get('plan') or []))))
                    d['camcal']['current_index'] = count if count < len(row.get('plan') or []) else None
            if e == 'tooloffset_command': d['command_seq'] = row.get('seq', d['command_seq'])
            if e == 'tooloffset_state':
                if row.get('phase') == 'triangulate' and not (row.get('triangulation') or {}).get('views'):
                    for key in ('triangulation', 'alignment', 'p0', 'p_final', 'plc_final', 'offset_xy', 'old_offset_xy', 'failure_code', 'failure_what', 'failure_where', 'failure_hint', 'next_camera_xyz'):
                        d['tooloffset'].pop(key, None)
                d['tooloffset'].update(row)
            if e == 'pickacc_state': d['pickacc'].update(row)
            if e == 'depthcal_state': d['depthcal'] = row
            if e == 'baseline_state': d['baseline'].update(row)
            if e == 'tooltip_start': d['tooltip'] = {}
            if e.startswith('tooltip_') and not e.startswith(('tooltip_preview_', 'tooltip_exposure_', 'tooltip_camera_', 'tooltip_preparation_', 'tooltip_focus_', 'tooltip_runout_')):
                d['tooltip'].update(row)
                if e in ('tooltip_detect','tooltip_offset','tooltip_reliability'): d['tooltip'].setdefault('history', []).append(dict(row))
            now = time.monotonic()
            if now - self.last >= .2 or (e == 'request' and row.get('plc_tool_id') is not None):
                self.write(); self.last = now
            elif self.timer is None:
                self.timer = threading.Timer(.2 - (now - self.last), self.flush)
                self.timer.daemon = True
                self.timer.start()

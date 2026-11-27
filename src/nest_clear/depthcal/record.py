"""按会话记录原始扫描、图像和已提交的 PLC 交互。"""

import csv
import json
import os

import cv2

from nest_clear.depth_config import METRICS
from nest_clear.paths import live_image_file


class DepthRecord:
    def __init__(self, events):
        self.events = events
        self.started = events.clock()
        self.directory = (events.directory / self.started.date().isoformat() / 'depthcal' /
                          f'{self.started:%H%M%S_%f}_{os.getpid()}')
        self.live = live_image_file(events.directory, 'depthcal_roi')

    def begin(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'roi').mkdir()
        self.live.unlink(missing_ok=True)
        with (self.directory / 'curves.csv').open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow(('label', 'pass', 'direction', 'z_cmd', 'z_actual', *METRICS,
                                         'exposure_us', 'roi_mean', 'roi_p99'))
        with (self.directory / 'arrivals.csv').open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow((
                'label', 'phase', 'seq',
                'expected_x', 'expected_y', 'expected_z',
                'actual_x', 'actual_y', 'actual_z',
                'delta_x', 'delta_y', 'delta_z',
                'max_abs_delta_mm', 'max_delta_axis',
                'ptp_x', 'ptp_y', 'ptp_z',
                'status', 'elapsed_s', 'samples',
            ))

    def arrival(self, entry):
        """追加记录每次 160 到位采样结果；目录未建立时跳过。"""
        if not self.directory.exists():
            return
        path = self.directory / 'arrivals.csv'
        if not path.exists():
            with path.open('w', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerow((
                    'label', 'phase', 'seq',
                    'expected_x', 'expected_y', 'expected_z',
                    'actual_x', 'actual_y', 'actual_z',
                    'delta_x', 'delta_y', 'delta_z',
                    'max_abs_delta_mm', 'max_delta_axis',
                    'ptp_x', 'ptp_y', 'ptp_z',
                    'status', 'elapsed_s', 'samples',
                ))
        expected = entry['expected'] if entry.get('expected') is not None else ('', '', '')
        actual = entry['actual']
        delta = entry['delta'] if entry.get('delta') is not None else ('', '', '')
        ptp = entry['ptp']
        max_abs = '' if entry.get('max_abs_delta_mm') is None else entry['max_abs_delta_mm']
        max_axis = '' if entry.get('max_delta_axis') is None else entry['max_delta_axis']
        with path.open('a', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow((
                entry['label'], entry['phase'], entry['seq'],
                *expected, *actual, *delta,
                max_abs, max_axis,
                *ptp,
                entry['status'], entry['elapsed_s'], entry['samples'],
            ))

    def image(self, name, frame):
        path = self.directory / f'{name}.png'
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f'无法写入 {path.name}')

    def roi(self, frame):
        self.live.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        ratio = min(1., 280 / max(h, w))
        preview = cv2.resize(frame, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                             interpolation=cv2.INTER_AREA)
        temp = self.live.with_suffix('.tmp.jpg')
        if not cv2.imwrite(str(temp), preview):
            raise OSError('无法写入 ROI 实时缩略图')
        temp.replace(self.live)

    def sample(self, row, frame, number):
        self.image(f"roi/{number:05d}_{row['label']}_{row['pass']}", frame)
        with (self.directory / 'curves.csv').open('a', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow(
                '' if row.get(key) is None else row[key]
                for key in ('label', 'pass', 'direction', 'z_cmd', 'z_actual', *METRICS,
                            'exposure_us', 'roi_mean', 'roi_p99')
            )
        self.roi(frame)

    def interaction(self, kind, row):
        """持久化实际请求和已提交应答，记录失败单独报告。"""
        try:
            with (self.directory / 'plc.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'event': kind, **row}, ensure_ascii=False, allow_nan=False) + '\n')
        except OSError as exc:
            self.events.emit('error', operation='depthcal_plc_record', reason=str(exc))

    def write(self, name, data):
        path = self.directory / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        temp.replace(path)

    def results(self, session):
        self.write('fits.json', {'metric': session.cfg['metric'], 'method': session.cfg['peak_method'],
                                 'dof_ratio': session.cfg['dof_ratio'],
                                 'points': session.results, 'feasibility': session.feasibility,
                                 'dof_summary': getattr(session, 'dof_summary', None),
                                 'arrival_summary': getattr(session, 'arrival_summary', None)})
        self.write('plane.json', {'initial': session.initial_plane, 'final': session.plane,
                                  'tolerance_mm': session.cfg['plane_tol_mm']})

    def finish(self, session, outcome, details):
        try:
            if outcome != 'done':
                self.results(session)
            fine_plans = {r['label']: r['fine_plan'] for r in session.results if r.get('fine_plan')}
            if session.label and getattr(session, 'fine_plan', None) and session.label not in fine_plans:
                fine_plans[session.label] = session.fine_plan
            point_exposures = {
                r['label']: {
                    'exposure_us': r.get('exposure_us'),
                    'exposure_mode': r.get('exposure_mode'),
                    'roi_mean': r.get('roi_mean'),
                    'roi_p99': r.get('roi_p99'),
                    'exposure_warning': r.get('exposure_warning'),
                }
                for r in session.results
            }
            self.write('session_record.json', {
                'start_time': self.started.isoformat(), 'end_time': self.events.clock().isoformat(),
                'outcome': outcome, 'diagnostic': details, 'configuration': session.cfg,
                'initial_exposure_us': getattr(session, 'initial_exposure_us', None),
                'initial_exposure_auto': getattr(session, 'initial_exposure_auto', None),
                'exposure_restored': getattr(session, 'exposure_restored', None),
                'exposure_restore_error': getattr(session, 'exposure_restore_error', None),
                'exposure_us': getattr(session, 'exposure_us', None), 'point_exposures': point_exposures,
                'plan': session.plan,
                'fine_plans': fine_plans,
                'dof_summary': getattr(session, 'dof_summary', None),
                'arrival_summary': getattr(session, 'arrival_summary', None),
                'requests': session.requests, 'replies': session.replies, 'samples': len(session.samples),
            })
        except (OSError, ValueError) as exc:
            self.events.emit('error', operation='depthcal_session_record', reason=str(exc))

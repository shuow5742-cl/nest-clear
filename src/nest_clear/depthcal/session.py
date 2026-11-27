"""测点居中、单向对焦扫描和多点平面判定。"""

from collections import deque
import math
import statistics

import numpy as np

from nest_clear.depth_config import METRICS
from nest_clear.guard import GuardError, format_pose
from nest_clear.tooloffset.detection import detect
from nest_clear.tooloffset.session import film_calibration
from nest_clear.vision.geometry import pixel_to_module
from .focus import (
    analyze_coarse_scan, find_side_blur, fit_coarse_fwhm, fit_peak, measure_dof, plan_fine_scan, scores,
)
from .layout import dof_summary, film_layout, fit_plane, repeatability


def ascending(bottom, top, step):
    """生成从 bottom 向上至 top 的步进 Z 队列（不含起点 bottom）。"""
    if top <= bottom + 1e-9:
        return deque()
    count = math.ceil((top - bottom) / step - 1e-10)
    return deque([min(top, bottom + index * step) for index in range(1, count + 1)])


def descending(top, bottom, step):
    count = math.ceil((top - bottom) / step - 1e-10)
    return deque([top - index * step for index in range(count)] + [bottom])


class Session:
    def __init__(self, config, calibration, record):
        self.record = record
        self.cfg = dict(config.depth_calibration)
        self.travel_is_fallback = self.cfg.get('travel') is None
        self.travel = self.cfg.get('travel') or config.travel
        self.cal = film_calibration(calibration, config.tool_offset['card_thickness_mm'],
                                    config.camera_calibration['board_thickness_mm'])
        self.z_min = max(self.travel[2][0], self.cfg['camera_z_min_mm'])
        self.z_max = self.travel[2][1]
        self.requests, self.replies = [], []
        self.abort_requested = False
        self.last_sample = None
        self.samples, self.results = [], []
        self.layout, primary, extra = film_layout()
        self.plan = list(self.cfg['point_order'] or primary)
        self.extra = [label for label in (self.cfg['extra_points'] or extra) if label not in self.plan]
        self.point_index = 0
        self.plane = self.initial_plane = self.feasibility = self.dof_summary = None
        self.actual = None
        self.all_arrivals = []
        self.arrival_summary = self._build_arrival_summary()
        self.reset_point()

    def reset_point(self):
        self.phase = 'center'
        self.label = None
        self.center_round = 1
        self.expected = None
        self.roi = None
        self.fits = []
        self.scan_rows = []
        self.coarse_up_rows = []
        self.coarse_up_limit = self.z_max
        self.coarse_down_limit = self.z_min
        self.coarse_upper_blurred = False
        self.coarse_upper_limited_by = None
        self.coarse_lower_limited_by = None
        self.coarse_bounds = None
        self._last_up_crop = None
        self.repeat = 0
        self.coarse_fit = None
        self.coarse_dof = None
        self.fine_plan = None
        self.result = None
        self.point_arrivals = []

    @property
    def frame_count(self):
        return self.cfg['frames_per_step'] if self.phase in ('coarse_up', 'coarse', 'fine') else 3

    @property
    def where(self):
        return f"{self.label or '初始靶点'} {self.message}"

    @property
    def message(self):
        return {
            'center': f'居中 第 {self.center_round} 轮',
            'coarse_up': '粗找 · 上行',
            'coarse_overshoot': '粗扫 · 抬升',
            'coarse': '粗扫 · 下行',
            'overshoot': f"细扫 第 {self.repeat}/{self.cfg['repeats_per_point']} 次 · 抬升",
            'fine': f"细扫 第 {self.repeat}/{self.cfg['repeats_per_point']} 次",
            'peak': '峰位全幅采图',
        }[self.phase]

    def guard(self, xyz):
        travel_name = ('下针深度标定行程（depth_calibration.travel，缺省回退全局 travel）'
                       if self.travel_is_fallback else '下针深度标定行程（depth_calibration.travel）')
        if len(xyz) != 3 or len(self.travel) != 3:
            raise GuardError('POSITION-TRAVEL',
                             f'坐标 {format_pose(xyz)} 无效，无法校验{travel_name}',
                             self.where, '检查下针深度标定行程（depth_calibration.travel）及目标坐标。')
        for axis, value, (lo, hi) in zip('XYZ', xyz, self.travel):
            if not math.isfinite(value) or not lo <= value <= hi:
                target = f'{value:.3f} mm' if math.isfinite(value) else repr(value)
                raise GuardError(
                    'POSITION-TRAVEL',
                    f'{axis} 轴目标值 {target} 超出{travel_name}允许范围 [{lo}, {hi}] mm（完整目标 {format_pose(xyz)}）',
                    self.where,
                    '检查菲林摆放位置、PLC 实时坐标与 config/nest_clear.yaml 中的下针深度标定行程（depth_calibration.travel）。',
                )
        if xyz[2] < self.z_min:
            raise GuardError('DEPTH-Z-LIMIT', f'相机 Z={xyz[2]:.6f} mm 低于安全下限 {self.z_min} mm',
                             self.where, '检查 PLC 到位坐标与现场核定的工具安全间隙。')

    def move(self, xyz):
        self.guard(xyz)
        # PLC 收到的是 REAL32。允许边界坐标向行程内部舍入。
        bounds = (*self.travel[:2], (self.z_min, self.z_max))
        encoded = []
        for value, (lo, hi) in zip(xyz, bounds):
            word = np.float32(value)
            if word < lo:
                word = np.nextafter(word, np.float32(np.inf))
            elif word > hi:
                word = np.nextafter(word, np.float32(-np.inf))
            encoded.append(float(word))
        self.expected = tuple(encoded)
        self.guard(self.expected)
        return self.expected

    def _arrival_phase(self):
        if self.phase == 'fine':
            return f'fine{self.repeat}'
        return self.phase

    @staticmethod
    def _build_point_arrival(arrivals):
        anomalies = [a for a in arrivals if a['status'] != 'ok']
        with_delta = [a for a in arrivals if a['max_abs_delta_mm'] is not None]
        if with_delta:
            worst = max(with_delta, key=lambda a: a['max_abs_delta_mm'])
            max_abs = worst['max_abs_delta_mm']
            max_signed = worst['max_delta_mm']
            max_axis = worst['max_delta_axis']
        else:
            max_abs = max_signed = max_axis = None
        elapsed_list = [a['elapsed_s'] for a in arrivals]
        return {
            'total': len(arrivals),
            'anomalies': len(anomalies),
            'over_tolerance': sum(1 for a in anomalies if a['over_tolerance']),
            'unstable': sum(1 for a in anomalies if a['unstable']),
            'max_abs_delta_mm': max_abs,
            'max_delta_mm': max_signed,
            'max_delta_axis': max_axis,
            'phases': list(dict.fromkeys(a['phase'] for a in anomalies)),
            'avg_elapsed_s': (sum(elapsed_list) / len(elapsed_list)) if elapsed_list else None,
            'max_elapsed_s': max(elapsed_list) if elapsed_list else None,
        }

    def _build_arrival_summary(self):
        anomalies = [a for a in self.all_arrivals if a['status'] != 'ok']
        with_delta = [a for a in self.all_arrivals if a['max_abs_delta_mm'] is not None]
        if with_delta:
            worst = max(with_delta, key=lambda a: a['max_abs_delta_mm'])
            max_abs = worst['max_abs_delta_mm']
            max_signed = worst['max_delta_mm']
            max_axis = worst['max_delta_axis']
            max_label = worst['point_label'] or worst['label']
        else:
            max_abs = max_signed = max_axis = max_label = None
        elapsed_list = [a['elapsed_s'] for a in self.all_arrivals]
        return {
            'total_arrivals': len(self.all_arrivals),
            'total_anomalies': len(anomalies),
            'over_tolerance_count': sum(1 for a in anomalies if a['over_tolerance']),
            'unstable_count': sum(1 for a in anomalies if a['unstable']),
            'anomaly_points': list(dict.fromkeys((a['point_label'] or a['label']) for a in anomalies)),
            'max_abs_delta_mm': max_abs,
            'max_delta_mm': max_signed,
            'max_delta_axis': max_axis,
            'max_delta_label': max_label,
            'position_tolerance_mm': float(self.cfg['position_tolerance_mm']),
            'avg_elapsed_s': (sum(elapsed_list) / len(elapsed_list)) if elapsed_list else None,
            'max_elapsed_s': max(elapsed_list) if elapsed_list else None,
        }

    def arrived(self, actual, arrival_info=None, *, task_id=None):
        self.guard(actual)
        actual_t = tuple(float(x) for x in actual)
        expected = self.expected
        if expected is not None:
            delta = tuple(float(a - e) for a, e in zip(actual_t, expected))
            pos_tol = self.cfg['position_tolerance_mm']
            over = any(abs(d) > pos_tol for d in delta)
            max_idx = max(range(3), key=lambda i: abs(delta[i]))
            max_delta_axis = 'XYZ'[max_idx]
            max_delta_mm = delta[max_idx]
            max_abs_delta_mm = abs(max_delta_mm)
        else:
            delta = None
            over = False
            max_delta_axis = None
            max_delta_mm = None
            max_abs_delta_mm = None

        stable = True if arrival_info is None else bool(arrival_info.get('stable', True))
        ptp = (0.0, 0.0, 0.0) if arrival_info is None else tuple(float(x) for x in arrival_info.get('ptp', (0.0, 0.0, 0.0)))
        elapsed_s = 0.0 if arrival_info is None else float(arrival_info.get('elapsed_s', 0.0))
        sample_count = 0 if arrival_info is None else int(arrival_info.get('samples', 0))

        if over and not stable:
            status = '超差+未稳定'
        elif over:
            status = '超差'
        elif not stable:
            status = '未稳定'
        else:
            status = 'ok'

        phase = self._arrival_phase()
        seq = len(self.all_arrivals) + 1
        label = self.label or '初始靶点'
        entry = {
            'seq': seq,
            'label': label,
            'point_label': self.label,
            'phase': phase,
            'expected': expected,
            'actual': actual_t,
            'delta': delta,
            'max_abs_delta_mm': max_abs_delta_mm,
            'max_delta_mm': max_delta_mm,
            'max_delta_axis': max_delta_axis,
            'ptp': ptp,
            'status': status,
            'over_tolerance': over,
            'unstable': not stable,
            'elapsed_s': elapsed_s,
            'samples': sample_count,
        }
        self.all_arrivals.append(entry)
        self.point_arrivals.append(entry)
        self.record.arrival(entry)
        if status != 'ok':
            delta_text = ('—' if delta is None else
                          f'({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) mm')
            emit_fields = {
                'label': label,
                'phase': phase,
                'seq': seq,
                'expected': list(expected) if expected is not None else None,
                'actual': list(actual_t),
                'delta': list(delta) if delta is not None else None,
                'max_abs_delta_mm': max_abs_delta_mm,
                'max_delta_mm': max_delta_mm,
                'max_delta_axis': max_delta_axis,
                'ptp': list(ptp),
                'type': status,
                'status': status,
                'elapsed_s': elapsed_s,
                'samples': sample_count,
                'what': f'{label} {phase} 到位{status}（偏差 {delta_text}）',
                'where': self.where,
                'hint': '查看 arrivals.csv 调整 settle_window_s / jitter_tol_mm / arrival_timeout_s 或检查 PLC 到位特性。',
                'reason': f'{label} {phase} 到位{status}：期望 {expected}，实际 {actual_t}，偏差 {delta}，极差 {ptp}',
            }
            if task_id is not None:
                emit_fields['task_id'] = task_id
            elif self.record.events.request_task is not None:
                emit_fields['task_id'] = self.record.events.request_task
            self.record.events.emit('depthcal_arrival_warning', **emit_fields)

        self.arrival_summary = self._build_arrival_summary()
        if self.result is not None and self.result.get('label') == self.label:
            self.result['arrival'] = self._build_point_arrival(self.point_arrivals)
        self.actual = actual_t
        return entry

    def advance(self, frames):
        if self.phase == 'center':
            return self.center(np.median(np.stack(frames), axis=0).astype(np.uint8))
        if self.phase == 'peak':
            return self.point_finished(frames[len(frames) // 2])
        return self.scan(frames)

    def center(self, frame):
        h, w = frame.shape[:2]
        uv = ((w - 1) / 2, (h - 1) / 2)
        targets = detect(frame)
        if not targets:
            raise GuardError('DEPTH-TARGET', '未识别到菲林圆环和 ArUco 靶点', self.where,
                             '检查初始位置、菲林方向、照明及相机焦距。')
        target = min(targets, key=lambda item: math.dist(item.ring_uv, uv))
        self.identify(target)
        self.label = target.label
        for item in self.point_arrivals:
            if item['point_label'] is None:
                item['point_label'] = self.label
        self.arrival_summary = self._build_arrival_summary()
        self.point_xy = pixel_to_module(*target.ring_uv, self.actual, self.cal)[:2]
        center_xy = pixel_to_module(*uv, self.actual, self.cal)[:2]
        x, y, z = self.actual
        self.center_offset = np.array(center_xy) - (x, y)
        if self.label != self.plan[self.point_index]:
            return self.locate(self.plan[self.point_index], z)
        if math.dist(target.ring_uv, uv) > self.cfg['center_tol_px']:
            if self.center_round >= self.cfg['center_max_iter']:
                raise GuardError('DEPTH-CENTER', '达到居中轮数上限，圆环仍偏离画面中心',
                                 self.where, '检查靶点、标定投影和 PLC 到位。')
            self.center_round += 1
            return self.move((x + self.point_xy[0] - center_xy[0], y + self.point_xy[1] - center_xy[1], z))
        u, v = target.ring_uv
        dx = math.dist(self.point_xy, pixel_to_module(u + 1, v, self.actual, self.cal)[:2])
        dy = math.dist(self.point_xy, pixel_to_module(u, v + 1, self.actual, self.cal)[:2])
        size = max(3, round(self.cfg['roi_mm'] / ((dx + dy) / 2)))
        left, top = round(u - (size - 1) / 2), round(v - (size - 1) / 2)
        if left < 0 or top < 0 or left + size > w or top + size > h:
            raise GuardError('DEPTH-ROI', f'固定 ROI {size} px 超出画面', self.where,
                             '减小 roi_mm，或检查居中和像素比例尺。')
        self.roi = np.s_[top:top + size, left:left + size]
        self.record.image(f'{self.label}_center', frame)
        crop = frame[self.roi]
        self.record.roi(crop)
        self.camera_xy = (x, y)
        return self._begin_coarse_up(crop, z)

    def _begin_coarse_up(self, crop, z):
        self._last_up_crop = crop
        self.coarse_up_rows = [{'z_actual': z, **scores(crop)}]
        self.coarse_upper_blurred = False
        self.coarse_upper_limited_by = None
        self.coarse_lower_limited_by = None
        self.coarse_bounds = None
        max_span = self.cfg.get('coarse_max_span_mm')
        up_limit = min(self.z_max, z + max_span) if max_span is not None else self.z_max
        self.coarse_up_limit = up_limit
        self.steps = ascending(z, up_limit, self.cfg['coarse_step_mm'])
        if self.steps:
            self.phase = 'coarse_up'
            return self.next_step()
        self.coarse_upper_limited_by = 'span' if (max_span is not None and abs(up_limit - self.z_max) > 1e-6) else 'z_max'
        return self._begin_coarse_down(z, seed_row=self.coarse_up_rows[-1])

    def _begin_coarse_down(self, z_top, seed_row=None):
        max_span = self.cfg.get('coarse_max_span_mm')
        down_limit = max(self.z_min, z_top - max_span) if max_span is not None else self.z_min
        self.coarse_down_limit = down_limit
        self.scan_rows = []
        self.steps = descending(z_top, down_limit, self.cfg['coarse_step_mm'])
        top_over = min(z_top + self.cfg['overshoot_mm'], self.z_max)
        if top_over > z_top + max(0.02, self.cfg['position_tolerance_mm']):
            self.phase = 'coarse_overshoot'
            return self.move((*self.camera_xy, top_over))
        if seed_row is not None:
            row = {
                'label': self.label, 'pass': 'coarse', 'direction': 'down',
                'z_cmd': float(seed_row['z_actual']), 'z_actual': float(seed_row['z_actual']),
                **{key: float(seed_row[key]) for key in METRICS},
            }
            self.scan_rows.append(row)
            self.samples.append(row)
            self.last_sample = row
            self.record.sample(row, self._last_up_crop, len(self.samples))
            if self.steps and abs(self.steps[0] - float(seed_row['z_actual'])) < 1e-6:
                self.steps.popleft()
        self.phase = 'coarse'
        self.overshoot_z = None
        return self.next_step()

    def identify(self, target):
        if self.label is None and target.label in self.plan:
            self.plan.remove(target.label)
            self.plan.insert(0, target.label)
        if self.label is not None and target.label != self.label:
            raise GuardError('DEPTH-TARGET', f'靶点从 {self.label} 变为 {target.label}',
                             self.where, '检查初始靶点和 PLC XY 到位坐标。')

    def locate(self, label, z):
        delta = self.layout[label] - self.layout[self.label]
        center_xy = self.center_offset
        x, y = (float(v) for v in (np.array(self.point_xy) + delta - center_xy))
        self.reset_point()
        self.label = label
        return self.move((x, y, z))

    def next_step(self):
        return self.move((*self.camera_xy, self.steps.popleft()))

    def begin_fine(self):
        self.repeat += 1
        self.phase = 'overshoot'
        self.scan_rows = []
        plan = self.fine_plan
        high = plan['high_mm']
        low = plan['low_mm']
        top = plan['top_mm']
        if high <= low:
            raise GuardError('DEPTH-FINE-RANGE', 'Z 行程不足以完成细扫换向与向下采样',
                             self.where, '检查粗扫峰位、细扫范围及可用 Z 行程。')
        self.steps = descending(high, low, plan['fine_step_mm'])
        return self.move((*self.camera_xy, top))

    def after_overshoot(self):
        self.phase = 'coarse' if self.phase == 'coarse_overshoot' else 'fine'
        self.overshoot_z = self.actual[2]
        return self.next_step()

    def scan(self, frames):
        z = self.actual[2]
        if self.phase == 'coarse_up':
            crops = [frame[self.roi] for frame in frames]
            metrics = [scores(crop) for crop in crops]
            primary = self.cfg['metric']
            midpoint = sorted(range(len(crops)), key=lambda i: metrics[i][primary])[len(crops) // 2]
            row = {'z_actual': z, **{key: statistics.median(item[key] for item in metrics) for key in METRICS}}
            self._last_up_crop = crops[midpoint]
            self.coarse_up_rows.append(row)
            self.record.roi(crops[midpoint])
            values = [item[primary] for item in self.coarse_up_rows]
            peak_idx = values.index(max(values))
            upper_blur = find_side_blur(
                values, list(range(peak_idx, len(values))), max(values), min(values),
                blur_contrast_ratio=self.cfg['blur_contrast_ratio'],
                defocus_ratio=self.cfg['defocus_ratio'],
                min_peak_contrast=self.cfg['min_peak_contrast'],
            )
            if upper_blur is not None or not self.steps:
                self.coarse_upper_blurred = upper_blur is not None
                if not self.coarse_upper_blurred:
                    max_span = self.cfg.get('coarse_max_span_mm')
                    self.coarse_upper_limited_by = (
                        'span' if (max_span is not None and abs(self.coarse_up_limit - self.z_max) > 1e-6) else 'z_max'
                    )
                return self._begin_coarse_down(z, seed_row=row)
            return self.next_step()
        previous_z = self.scan_rows[-1]['z_actual'] if self.scan_rows else None
        if previous_z is None and self.phase == 'fine':
            previous_z = self.overshoot_z
        elif previous_z is None and self.phase == 'coarse':
            previous_z = getattr(self, 'overshoot_z', None)
        if previous_z is not None and z >= previous_z:
            raise GuardError('DEPTH-DIRECTION', '实测 Z 没有继续下降', self.where,
                             '核对 PLC 到位精度和细扫步长。')
        crops = [frame[self.roi] for frame in frames]
        metrics = [scores(crop) for crop in crops]
        primary = self.cfg['metric']
        midpoint = sorted(range(len(crops)), key=lambda i: metrics[i][primary])[len(crops) // 2]
        row = {'label': self.label, 'pass': 'coarse' if self.phase == 'coarse' else self.repeat,
               'direction': 'down', 'z_cmd': self.expected[2], 'z_actual': z,
               **{key: statistics.median(item[key] for item in metrics) for key in METRICS}}
        self.scan_rows.append(row)
        self.samples.append(row)
        self.last_sample = row
        self.record.sample(row, crops[midpoint], len(self.samples))
        values = [item[primary] for item in self.scan_rows]
        if self.phase == 'coarse':
            peak_idx = values.index(max(values))
            lower_blur = find_side_blur(
                values, list(range(peak_idx, len(values))), max(values), min(values),
                blur_contrast_ratio=self.cfg['blur_contrast_ratio'],
                defocus_ratio=self.cfg['defocus_ratio'],
                min_peak_contrast=self.cfg['min_peak_contrast'],
            )
            if self.steps and lower_blur is None:
                return self.next_step()
            if lower_blur is None:
                max_span = self.cfg.get('coarse_max_span_mm')
                self.coarse_lower_limited_by = (
                    'span' if (max_span is not None and abs(self.coarse_down_limit - self.z_min) > 1e-6) else 'z_min'
                )
            analysis = analyze_coarse_scan(
                self.scan_rows, primary, self.cfg, self.where,
                upper_blurred_hint=self.coarse_upper_blurred,
                upper_limited_by=self.coarse_upper_limited_by,
                lower_limited_by=self.coarse_lower_limited_by,
            )
            self.coarse_bounds = analysis
            dof_ratio = self.cfg['dof_ratio']
            coarse_fit, _ = fit_coarse_fwhm(
                self.scan_rows, primary, self.cfg['peak_method'], self.where, dof_ratio=dof_ratio,
            )
            self.coarse_fit = coarse_fit
            if coarse_fit is not None:
                self.coarse_dof = {
                    k: coarse_fit[k]
                    for k in ('dof_mm', 'dof_gauss_mm', 'dof_ratio', 'dof_z_lower', 'dof_z_upper', 'dof_covered')
                    if k in coarse_fit
                }
            else:
                ordered = sorted(self.scan_rows, key=lambda item: item['z_actual'])
                z_arr = np.array([item['z_actual'] for item in ordered])
                v_arr = np.array([item[primary] for item in ordered])
                self.coarse_dof = {**measure_dof(z_arr, v_arr, int(np.argmax(v_arr)), dof_ratio), 'dof_gauss_mm': None}
            self.coarse_peak = coarse_fit['z_peak'] if coarse_fit else analysis['z_peak_coarse']
            coarse_fwhm = coarse_fit['fwhm'] if coarse_fit else None
            self.fine_plan = plan_fine_scan(
                self.cfg, analysis['blur_low_z'], analysis['blur_high_z'], self.z_min, self.z_max,
                blur_low_reached=analysis['blur_low_reached'],
                blur_high_reached=analysis['blur_high_reached'],
                coarse_peak=self.coarse_peak,
                coarse_fwhm=coarse_fwhm,
                limit_notes=analysis['limit_notes'],
            )
            return self.begin_fine()
        if self.steps:
            return self.next_step()
        dof_ratio = self.cfg['dof_ratio']
        fit = fit_peak(self.scan_rows, primary, self.cfg['peak_method'], self.where, dof_ratio=dof_ratio)
        comparisons = {primary: fit}
        for metric in METRICS:
            if metric != primary:
                try:
                    comparisons[metric] = fit_peak(self.scan_rows, metric, self.cfg['peak_method'], self.where, dof_ratio=dof_ratio)
                except GuardError as exc:
                    comparisons[metric] = {'failure': exc.what}
        self.fits.append({'repeat': self.repeat, **fit, 'metrics': comparisons})
        if self.repeat < self.cfg['repeats_per_point']:
            return self.begin_fine()
        peaks = [item['z_peak'] for item in self.fits]
        insufficient = [item['repeat'] for item in self.fits if not item.get('quality_ok', True)]
        covered_dofs = [item['dof_mm'] for item in self.fits if item.get('dof_covered') and item.get('dof_mm') is not None]
        uncovered_dofs = [item['repeat'] for item in self.fits if not item.get('dof_covered')]
        self.result = {
            'label': self.label, 'xy': [float(v) for v in self.point_xy], 'fits': self.fits,
            'coarse_fit': self.coarse_fit, 'coarse_dof': self.coarse_dof,
            'coarse_bounds': self.coarse_bounds, 'fine_plan': self.fine_plan,
            'insufficient_repeats': insufficient,
            'z_peak': statistics.mean(peaks), 'range': max(peaks) - min(peaks),
            'std': statistics.stdev(peaks) if len(peaks) > 1 else None,
            'dof_mm': statistics.mean(covered_dofs) if covered_dofs else None,
            'dof_range_mm': (max(covered_dofs) - min(covered_dofs)) if len(covered_dofs) >= 2 else None,
            'dof_covered_count': len(covered_dofs),
            'dof_uncovered': len(uncovered_dofs),
            'dof_uncovered_repeats': uncovered_dofs,
            'arrival': self._build_point_arrival(self.point_arrivals),
        }
        self.results.append(self.result)
        self.phase = 'peak'
        return self.move((*self.camera_xy, self.result['z_peak']))

    def point_finished(self, frame):
        self.record.image(f'{self.label}_peak', frame)
        self.record.roi(frame[self.roi])
        self.result['peak_image_z'] = self.actual[2]
        self.point_index += 1
        if self.point_index == len(self.plan):
            self.plane = fit_plane(self.results)
            if self.initial_plane is None:
                self.initial_plane = self.plane
                if self.plane['max_residual'] > self.cfg['plane_tol_mm']:
                    slots = self.cfg['max_points'] - len(self.plan)
                    self.plan.extend(self.extra[:slots])
            else:
                delta = self.plane['max_residual'] - self.initial_plane['max_residual']
                self.plane['after_extra'] = ('加测后残差下降' if delta < -1e-6 else
                                             '加测后残差增大' if delta > 1e-6 else '加测后残差不变')
        if self.point_index == len(self.plan):
            self.feasibility = repeatability(self.results, self.cfg['repeat_range_tol_mm'])
            self.dof_summary = dof_summary(self.results, self.cfg['dof_ratio'])
            return None
        return self.locate(self.plan[self.point_index], self.result['z_peak'])

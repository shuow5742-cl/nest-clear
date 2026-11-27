"""测点居中、单向对焦扫描和多点平面判定。"""

from collections import deque
import math
import statistics

import numpy as np

from nest_clear.depth_config import METRICS
from nest_clear.guard import GuardError, format_pose
from nest_clear.plan.scan_path import guard_position
from nest_clear.tooloffset.detection import detect
from nest_clear.tooloffset.session import film_calibration
from nest_clear.vision.geometry import pixel_to_module
from .focus import fit_coarse_fwhm, fit_peak, peak_index, plan_fine_scan, scores
from .layout import film_layout, fit_plane, repeatability


def descending(top, bottom, step):
    count = math.ceil((top - bottom) / step - 1e-10)
    return deque([top - index * step for index in range(count)] + [bottom])


class Session:
    def __init__(self, config, calibration, record):
        self.record = record
        self.cfg = dict(config.depth_calibration)
        self.travel = config.travel
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
        self.plane = self.initial_plane = self.feasibility = None
        self.actual = None
        self.reset_point()

    def reset_point(self):
        self.phase = 'center'
        self.label = None
        self.center_round = 1
        self.expected = None
        self.roi = None
        self.fits = []
        self.scan_rows = []
        self.repeat = 0
        self.coarse_fit = None
        self.fine_plan = None

    @property
    def frame_count(self):
        return self.cfg['frames_per_step'] if self.phase in ('coarse', 'fine') else 3

    @property
    def where(self):
        return f"{self.label or '初始靶点'} {self.message}"

    @property
    def message(self):
        return {
            'center': f'居中 第 {self.center_round} 轮',
            'coarse': '粗扫',
            'overshoot': f"细扫 第 {self.repeat}/{self.cfg['repeats_per_point']} 次 · 抬升",
            'fine': f"细扫 第 {self.repeat}/{self.cfg['repeats_per_point']} 次",
            'peak': '峰位全幅采图',
        }[self.phase]

    def guard(self, xyz):
        guard_position(xyz, self.travel, self.where)
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

    def arrived(self, actual):
        self.guard(actual)
        if self.expected and any(abs(a - b) > self.cfg['position_tolerance_mm']
                                 for a, b in zip(actual, self.expected)):
            raise GuardError('DEPTH-ARRIVAL',
                             f'相机到位超差，目标 {format_pose(self.expected)}，实际 {format_pose(actual)}',
                             self.where, '检查 PLC 到位时序和 position_tolerance_mm。')
        self.actual = actual

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
        self.record.roi(frame[self.roi])
        self.camera_xy = (x, y)
        self.phase = 'coarse'
        radius = self.cfg['coarse_range_mm']
        self.steps = descending(min(z + radius, self.z_max), max(z - radius, self.z_min),
                                self.cfg['coarse_step_mm'])
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
        self.phase = 'fine'
        self.overshoot_z = self.actual[2]
        return self.next_step()

    def scan(self, frames):
        z = self.actual[2]
        previous_z = self.scan_rows[-1]['z_actual'] if self.scan_rows else None
        if previous_z is None and self.phase == 'fine':
            previous_z = self.overshoot_z
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
        index = values.index(max(values))
        if self.phase == 'coarse':
            threshold = self.cfg['defocus_ratio'] * values[index]
            defocused = len(values) - index >= 3 and max(values[-2:]) < threshold
            if self.steps and not defocused:
                return self.next_step()
            z_values = [item['z_actual'] for item in self.scan_rows]
            peak_index(z_values, values, self.where)
            coarse_fit, coarse_error = fit_coarse_fwhm(self.scan_rows, primary, self.cfg['peak_method'], self.where)
            self.coarse_fit = coarse_fit
            self.coarse_peak = coarse_fit['z_peak'] if coarse_fit else z_values[index]
            coarse_fwhm = coarse_fit['fwhm'] if coarse_fit else None
            self.fine_plan = plan_fine_scan(self.cfg, coarse_fwhm, self.coarse_peak, self.z_min, self.z_max, coarse_error)
            return self.begin_fine()
        if self.steps:
            return self.next_step()
        fit = fit_peak(self.scan_rows, primary, self.cfg['peak_method'], self.where)
        comparisons = {primary: fit}
        for metric in METRICS:
            if metric != primary:
                try:
                    comparisons[metric] = fit_peak(self.scan_rows, metric, self.cfg['peak_method'], self.where)
                except GuardError as exc:
                    comparisons[metric] = {'failure': exc.what}
        self.fits.append({'repeat': self.repeat, **fit, 'metrics': comparisons})
        if self.repeat < self.cfg['repeats_per_point']:
            return self.begin_fine()
        peaks = [item['z_peak'] for item in self.fits]
        insufficient = [item['repeat'] for item in self.fits if not item.get('quality_ok', True)]
        self.result = {'label': self.label, 'xy': [float(v) for v in self.point_xy], 'fits': self.fits,
                       'coarse_fit': self.coarse_fit, 'fine_plan': self.fine_plan,
                       'insufficient_repeats': insufficient,
                       'z_peak': statistics.mean(peaks), 'range': max(peaks) - min(peaks),
                       'std': statistics.stdev(peaks) if len(peaks) > 1 else None}
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
            return None
        return self.locate(self.plan[self.point_index], self.result['z_peak'])

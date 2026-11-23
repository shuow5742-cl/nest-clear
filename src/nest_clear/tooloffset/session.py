"""工具偏移标定会话计算，不执行 PLC 或界面 I/O。"""
import csv
from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from nest_clear.vision.geometry import pixel_to_module
from nest_clear.guard import GuardError

DEFAULTS = dict(center_tol_mm=1.0, outlier_mm=.1, min_points=5, max_points=12,
                se_tol_mm=.02, edge_warn_mm=.05, card_thickness_mm=.10,
                ring_cross_tol_mm=.08, settle_window_s=.5, settle_samples=6,
                jitter_tol_mm=.02, point_order=['T02','T05','T26','T29','T12','T19','T03','T28','T10','T21'],
                edge_targets=['T12'], prior_offset_mm=[29.3, 24.2])


def nominal_targets():
    path = Path(__file__).resolve().parents[3] / 'docs/pick_test_film/targets.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        return {int(row['index']): np.array([float(row['x_mm']), float(row['y_mm'])]) for row in csv.DictReader(stream)}


def fit_similarity(nominal, measured):
    """最小二乘求二维正向相似变换。"""
    a, b = np.asarray(nominal, dtype=float), np.asarray(measured, dtype=float)
    if len(a) < 2 or a.shape != b.shape or a.shape[1:] != (2,):
        raise ValueError('至少需要两个目标拟合标定卡')
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    src, dst = a - ac, b - bc
    den = float(np.sum(src * src))
    if den < 1e-12:
        raise ValueError('标定卡目标位置重合')
    p = float(np.sum(src * dst) / den)
    q = float(np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]) / den)
    matrix = np.array([[p, -q], [q, p]])
    return matrix, bc - matrix @ ac, math.hypot(p, q)


def stats(samples):
    valid = [s['offset_xy_mm'] for s in samples if not s['outlier']]
    if not valid:
        return None, None, None
    x = np.asarray(valid)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1) if len(x) > 1 else np.array([math.inf, math.inf])
    return mean, std, std / math.sqrt(len(x))


def film_calibration(calibration, card_thickness_mm, board_thickness_mm):
    """将相机标定板平面平移到菲林片表面。"""
    normal = calibration.plane_normal
    if abs(normal[2]) < 1e-6:
        raise ValueError('相机标定平面法向近乎水平，无法确定板面上下方向')
    upward = normal if normal[2] > 0 else -normal
    shift = card_thickness_mm - board_thickness_mm
    return replace(calibration, plane_point=calibration.plane_point + shift * upward)


class Session:
    def __init__(self, calibration, start, travel, config=None, mode='center', tool=1, board_thickness_mm=0., prior_offset=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.cal = film_calibration(calibration, self.cfg['card_thickness_mm'], board_thickness_mm)
        self.start, self.travel, self.mode, self.tool = tuple(start), travel, mode, tool
        self.nominal = nominal_targets()
        self.phase = 'overview'
        self.predicted = {}
        self.queue = []
        self.prior_offset = np.asarray(prior_offset if prior_offset is not None else self.cfg['prior_offset_mm'], dtype=float)
        self.camera_clamped = False
        self.expected_uv = None
        self.target_uv = None
        self.samples, self.edges, self.skipped = [], [], []
        self.measured = {}
        self.retested = set()
        self.pending_retests = set()
        self.current = None
        self.retries = 0
        self.target_pose = self.start
        self.edge_queue = []
        self.failure = None
        self.image_size = None

    def _fail(self, reason):
        self.phase, self.failure = 'failed', reason
        return 'failed'

    def _within(self, p):
        return all(lo <= v <= hi for v, (lo, hi) in zip(p, self.travel))

    def _pose_for(self, target_xy, uv):
        p = pixel_to_module(*uv, self.start, self.cal)
        return (self.start[0] + target_xy[0] - p[0], self.start[1] + target_xy[1] - p[1], self.start[2])

    def _clamp_pose(self, pose):
        return (float(np.clip(pose[0], *self.travel[0])), float(np.clip(pose[1], *self.travel[1])), self.start[2])

    def _project(self, xy, pose):
        uv = np.array([self.cal.K[0,2], self.cal.K[1,2]], dtype=float)
        try:
            for _ in range(20):
                point = np.asarray(pixel_to_module(*uv, pose, self.cal)[:2])
                delta = np.asarray(xy) - point
                if np.linalg.norm(delta) <= 1e-4:
                    return uv
                jac = np.column_stack([np.asarray(pixel_to_module(*(uv + shift), pose, self.cal)[:2]) - point
                                       for shift in ([1., 0.], [0., 1.])])
                uv += np.linalg.solve(jac, delta)
        except (ValueError, np.linalg.LinAlgError, GuardError):
            pass
        return None

    def _in_image(self, uv):
        w, h = self.image_size
        return .1*w <= uv[0] <= .9*w and .1*h <= uv[1] <= .9*h

    def _move_next(self):
        mean, _, se = stats(self.samples)
        count = sum(not s['outlier'] for s in self.samples)
        if len(self.samples) < self.cfg['max_points']:
            measured = {int(s['target'][1:]) for s in self.samples}
            skipped = {int(s['target'][1:]) for s in self.skipped if 'corner' not in s}
            candidates = []
            indices = sorted(self.pending_retests) if self.pending_retests else sorted(set(self.predicted) - measured - skipped)
            for index in indices:
                pos = self.predicted[index]
                pose = self._pose_for(pos, (self.cal.K[0,2], self.cal.K[1,2]))
                clamped = self._clamp_pose(pose)
                uv = self._project(pos, clamped) if clamped != pose else np.array([self.cal.K[0,2], self.cal.K[1,2]])
                if not self._within(clamped) or uv is None or not self._in_image(uv):
                    reason = '相机不可达'
                elif mean is not None and not self._within((pos[0]+mean[0], pos[1]+mean[1], clamped[2])):
                    reason = '工具不可达'
                else:
                    candidates.append((index, clamped, uv, clamped != pose))
                    continue
                self.skipped.append({'target': f'T{index:02d}', 'reason': reason})
                self.pending_retests.discard(index)
            if candidates:
                if measured:
                    def score(item):
                        return min(np.linalg.norm(self.predicted[item[0]] - self.predicted[i]) for i in measured)
                else:
                    def score(item):
                        tool = self.predicted[item[0]] + self.prior_offset
                        return min(tool[0]-self.travel[0][0], self.travel[0][1]-tool[0],
                                   tool[1]-self.travel[1][0], self.travel[1][1]-tool[1])
                index, pose, uv, clamped = max(candidates, key=score)
                self.current, self.target_pose, self.expected_uv, self.camera_clamped = index, pose, uv, clamped
                self.retries, self.phase = 0, 'center'
                return 'continue'
            if self.pending_retests:
                return self._move_next()
            if indices and measured and not candidates:
                return self._move_next()
        if count < self.cfg['min_points']:
            skipped_count = len({s['target'] for s in self.skipped if 'corner' not in s})
            return self._fail(f'可用目标不足：已测有效 {count} 个，需 {self.cfg["min_points"]} 个；32 个目标中 {skipped_count} 个因相机或工具超出行程被跳过。请把菲林片往行程中部挪动后重试')
        return self._fail(f'多点不一致：剩余可用目标已用完，有效 {count} 个，标准误 ({se[0]:.4f},{se[1]:.4f}) 超过 se_tol_mm，检查人工对准、卡片固定和震动')

    def _next_edge(self):
        while self.edge_queue:
            index, corner = self.edge_queue.pop(0)
            w, h = self.image_size
            u, v = ((.12 if corner in (0, 2) else .88) * w, (.12 if corner in (0, 1) else .88) * h)
            target = self.measured[index]
            pose = self._pose_for(target, (u, v))
            if not self._within(pose):
                self.skipped.append({'target': f'T{index:02d}', 'corner': corner, 'reason': '边缘验证相机坐标超出行程'})
                continue
            self.current, self.corner, self.target_pose, self.phase = index, corner, pose, 'edge'
            return 'continue'
        self.phase = 'confirm'
        return 'confirm'

    def image(self, targets, pose, image_size):
        self.image_size = image_size
        targets = {t.index: t for t in targets}
        if self.phase == 'overview':
            if len(targets) < 2:
                return self._fail('未找到标定卡：标定卡不在视野内/摆放不当/光源问题')
            points = {i: pixel_to_module(*t.ring_uv, pose, self.cal)[:2] for i, t in targets.items() if i in self.nominal}
            if len(points) < 2:
                return self._fail('未找到标定卡：标定卡不在视野内/摆放不当/光源问题')
            matrix, trans, scale = fit_similarity([self.nominal[i] for i in points], list(points.values()))
            if abs(scale - 1) > .03:
                return self._fail('标定卡尺寸不符，检查打印缩放是否为实际大小')
            self.predicted = {i: matrix @ p + trans for i, p in self.nominal.items()}
            return self._move_next()
        target = targets.get(self.current)
        if target is None:
            return self._fail(f'无法识别目标 T{self.current:02d}，检查标定卡、光照和相机位置')
        if target.cross_error_mm > 2 * self.cfg['ring_cross_tol_mm']:
            return self._fail('目标识别质量差：标定卡打印/摆放/光源问题')
        p = np.asarray(pixel_to_module(*target.ring_uv, pose, self.cal)[:2])
        if self.phase == 'edge':
            diff = p - self.measured[self.current]
            self.edges.append({'target': f'T{self.current:02d}', 'corner': self.corner,
                               'delta_xy_mm': diff.tolist(), 'distance_mm': float(np.linalg.norm(diff))})
            return self._next_edge()
        if self.phase != 'center':
            raise ValueError('当前阶段不能接收图像')
        expected = self.expected_uv if self.expected_uv is not None else (self.cal.K[0,2], self.cal.K[1,2])
        principal = np.asarray(pixel_to_module(*expected, pose, self.cal)[:2])
        error = p - principal
        if np.linalg.norm(error) > self.cfg['center_tol_mm']:
            corrected = self._clamp_pose((pose[0] + error[0], pose[1] + error[1], self.start[2]))
            if np.allclose(corrected, pose, atol=1e-9, rtol=0):
                self.camera_clamped = True
                if not self._in_image(target.ring_uv):
                    return self._fail('目标超出画面中央 80% 范围')
            elif self.retries >= 2:
                return self._fail(f'目标 T{self.current:02d} 对中重试超限')
            else:
                self.retries += 1
                self.target_pose = corrected
                self.camera_clamped = self.camera_clamped or corrected != (pose[0] + error[0], pose[1] + error[1], self.start[2])
                return 'continue'
        if self.camera_clamped and not self._in_image(target.ring_uv):
            return self._fail('目标超出画面中央 80% 范围')
        self.target_uv = list(target.ring_uv)
        self.measured[self.current] = p
        self.phase = 'tool'
        return 'tool'

    def add_sample(self, actual):
        if self.phase != 'tool':
            raise ValueError('当前不等待工具到位')
        target = self.measured[self.current]
        offset = np.asarray(actual[:2]) - target
        mean, _, _ = stats(self.samples)
        distance = float(np.linalg.norm(offset - mean)) if mean is not None else 0.
        outlier = mean is not None and distance > self.cfg['outlier_mm']
        self.samples.append({'target': f'T{self.current:02d}', 'target_xy_mm': target.tolist(),
                             'tool_xyz_mm': list(actual), 'offset_xy_mm': offset.tolist(),
                             'distance_from_mean_mm': distance, 'outlier': bool(outlier),
                             'camera_clamped': bool(self.camera_clamped), 'target_uv': self.target_uv})
        if self.current in self.pending_retests:
            self.pending_retests.remove(self.current)
        elif outlier and self.current not in self.retested:
            self.retested.add(self.current)
            self.pending_retests.add(self.current)
        mean, std, se = stats(self.samples)
        count = sum(not x['outlier'] for x in self.samples)
        if count >= self.cfg['min_points'] and np.all(se <= self.cfg['se_tol_mm']) and not self.pending_retests:
            if self.mode == 'center_edge':
                indices = [int(x[1:]) for x in self.cfg['edge_targets'] if int(x[1:]) in self.measured][:2]
                if not indices:
                    indices = [int(next(s['target'] for s in self.samples if not s['outlier'])[1:])]
                self.edge_queue = [(i, c) for i in indices for c in range(4)]
                return self._next_edge()
            self.phase = 'confirm'
            return 'confirm'
        return self._move_next()

    def result(self):
        mean, std, se = stats(self.samples)
        return {'offset_xy_mm': mean.tolist(), 'n': sum(not s['outlier'] for s in self.samples),
                'std': std.tolist(), 'se': se.tolist(), 'samples': self.samples,
                'edge_validation': self.edges, 'mode': self.mode, 'skipped': self.skipped}

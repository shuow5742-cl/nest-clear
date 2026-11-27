"""测点居中、单向对焦扫描和多点平面判定。"""

from collections import deque
import math
import statistics

import cv2
import numpy as np

from nest_clear.camcal.session import build_charuco_board, detect_charuco_corners
from nest_clear.depth_config import METRICS
from nest_clear.guard import GuardError, format_pose
from nest_clear.tooloffset.detection import detect
from nest_clear.tooloffset.session import film_calibration
from nest_clear.vision.geometry import pixel_to_module
from .exposure import load_depthcal_exposure
from .focus import (
    EXPOSURE_MAX_SATURATED_FRACTION, analyze_coarse_scan, find_side_blur, fit_coarse_fwhm, fit_peak,
    measure_dof, next_auto_exposure, plan_fine_scan, roi_brightness, scores,
)
from .layout import (
    charuco_board_outer_quad, charuco_corner_id, charuco_layout, charuco_safe_quad,
    corner_labels_for_plan, dof_summary, film_layout, fit_board_to_module, fit_plane,
    point_in_board_safe_zone, repeatability,
)

# 首两点连续失败提前终止规则开关（规则 a）：前两个计划点（含其补测）均失败时立即回 999。
ENABLE_FIRST_TWO_FAIL_ABORT = True

# 允许触发单点失败跳过/补测的业务错误码（通信、行程/Z下限、曝光改动、放弃等仍立即 999）。
SKIPPABLE_POINT_CODES = frozenset({
    'DEPTH-CONTRAST',
    'DEPTH-PEAK-BOUNDARY',
    'DEPTH-FIT',
    'DEPTH-TARGET',
    'DEPTH-CENTER',
    'DEPTH-ROI',
    'DEPTH-DARK',
    'DEPTH-FINE-RANGE',
    'DEPTH-SAFE-ZONE',
})


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
        self.configured_target = self.cfg.get('target', 'auto')
        self.board_thickness_mm = float(config.camera_calibration['board_thickness_mm'])
        self.camcal_cfg = dict(config.camera_calibration)
        self.film_cal = film_calibration(calibration, config.tool_offset['card_thickness_mm'],
                                         self.board_thickness_mm)
        self.board_cal = calibration
        self.target_type = None if self.configured_target == 'auto' else self.configured_target
        self.cal = self.board_cal if self.target_type == 'charuco' else self.film_cal
        self.z_min = max(self.travel[2][0], self.cfg['camera_z_min_mm'])
        self.z_max = self.travel[2][1]
        self.requests, self.replies = [], []
        self.abort_requested = False
        self.last_sample = None
        self.samples, self.results = [], []
        self.board_fit = None
        self.board_outer_quad = None
        self.point_positions = {}
        self._charuco_board = None
        self._charuco_detector = None
        self._charuco_marker_detector = None
        self._charuco_board_points = None
        if self.configured_target in ('auto', 'film'):
            self.layout, primary, extra = film_layout()
            self.plan = list(self.cfg['point_order'] or primary)
            self.extra = [label for label in (self.cfg['extra_points'] or extra) if label not in self.plan]
            self.corner_labels = corner_labels_for_plan(self.layout, self.plan)
            self.plan_slots = {label: idx for idx, label in enumerate(self.plan)}
        else:
            self.layout = {}
            self.plan = []
            self.extra = []
            self.corner_labels = set()
            self.plan_slots = {}
        self.slot_outcomes = {}
        self.retest_parent = {}
        self.attempted_labels = set()
        self.failed_points = []
        self.skip_summary = self._build_skip_summary()
        self.ref_label = None
        self.ref_xy = None
        self.initial_z = None
        self.last_center_image = None
        self.current_retest_of = None
        self.point_index = 0
        self.plane = self.initial_plane = self.feasibility = self.dof_summary = None
        self.actual = None
        self.all_arrivals = []
        self.arrival_summary = self._build_arrival_summary()
        exp_state = load_depthcal_exposure(record.events.directory, self.cfg)
        self.desired_exposure_mode = exp_state['mode']
        self.desired_manual_exposure_us = exp_state['manual_exposure_us']
        self.initial_exposure_us = None
        self.initial_exposure_auto = None
        self.exposure_restored = False
        self.exposure_restore_error = None
        self.exposure_us = None
        self.reset_point()

    @property
    def target_type_label(self):
        if self.target_type == 'film':
            return '菲林'
        if self.target_type == 'charuco':
            return f'ChArUco 标定板（厚 {self.board_thickness_mm:.2f}mm）'
        return '自动检测（待识别）'

    def _ensure_charuco(self):
        if self._charuco_detector is None:
            cfg = self.camcal_cfg
            self._charuco_board = build_charuco_board(
                cfg['cols'], cfg['rows'], cfg['square_mm'], cfg['marker_mm'], cfg['dictionary'],
            )
            self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)
            self._charuco_marker_detector = cv2.aruco.ArucoDetector(self._charuco_board.getDictionary())
            self._charuco_board_points = np.asarray(self._charuco_board.getChessboardCorners(), dtype=float)

    def _detect_film_targets(self, frame):
        targets = detect(frame)
        if not targets:
            return []
        self._ensure_charuco()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        m5, _, _ = self._charuco_marker_detector.detectMarkers(gray)
        if not m5:
            return targets
        polys = [m.reshape(4, 2).astype(np.float32) for m in m5]
        filtered = []
        for item in targets:
            mc = np.mean(item.marker, axis=0)
            pt = (float(mc[0]), float(mc[1]))
            if any(cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in polys):
                continue
            filtered.append(item)
        return filtered

    def set_desired_exposure(self, mode, manual_exposure_us):
        """更新期望曝光设置；若当前点已进入扫描则保持本点锁定，下一测点生效。"""
        self.desired_exposure_mode = mode
        self.desired_manual_exposure_us = manual_exposure_us

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
        self._ae = None
        self.point_exposure_us = None
        self.point_exposure_mode = getattr(self, 'desired_exposure_mode', 'auto')
        self.point_roi_mean = None
        self.point_roi_p99 = None
        self.point_roi_sat_frac = None
        self.point_exposure_warning = None
        self.live_roi_mean = None
        self.live_roi_p99 = None
        self.exposure_locked = False
        self.repeat = 0
        self.coarse_fit = None
        self.coarse_dof = None
        self.fine_plan = None
        self.result = None
        self.point_arrivals = []
        self.current_retest_of = None

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
            'exposure': '测点自动曝光' if self.point_exposure_mode == 'auto' else '应用手动曝光',
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

    def _target_module_xy(self, label):
        if label not in self.layout:
            return None
        if self.ref_xy is not None and self.ref_label is not None:
            return np.array(self.ref_xy, dtype=float) + (self.layout[label] - self.layout[self.ref_label])
        return None

    def _target_camera_xy(self, label):
        module_xy = self._target_module_xy(label)
        if module_xy is None or getattr(self, 'center_offset', None) is None:
            return None
        xy = module_xy - self.center_offset
        return (float(xy[0]), float(xy[1]))

    def _in_xy_travel(self, xy):
        if xy is None:
            return False
        x, y = xy
        (x_lo, x_hi), (y_lo, y_hi) = self.travel[:2]
        return math.isfinite(x) and math.isfinite(y) and x_lo <= x <= x_hi and y_lo <= y <= y_hi

    def _camera_xy_to_module_xy(self, cam_xy):
        if cam_xy is None or getattr(self, 'center_offset', None) is None:
            return None
        return np.array([float(cam_xy[0]), float(cam_xy[1])], dtype=float) + np.asarray(self.center_offset, dtype=float)

    def _in_board_safe_zone(self, cam_xy, *, tol_mm=0.05):
        if self.target_type != 'charuco' or self.board_outer_quad is None:
            return True
        mod_xy = self._camera_xy_to_module_xy(cam_xy)
        if mod_xy is None or not np.isfinite(mod_xy).all():
            return False
        keepout = float(self.cfg.get('board_keepout_mm', 5.0))
        return point_in_board_safe_zone(mod_xy, self.board_outer_quad, keepout, tol_mm=tol_mm)

    def _is_valid_target(self, label):
        cam_xy = self._target_camera_xy(label)
        return self._in_xy_travel(cam_xy) and self._in_board_safe_zone(cam_xy)

    def _select_backup_target(self, failed_label):
        """选择离失败点最近、未测过/未计划且在 depth_calibration.travel 与安全区内的备用测点。"""
        if failed_label not in self.layout:
            return None
        queued = set(self.plan[self.point_index + 1:])
        candidates = [
            cand for cand in self.layout
            if cand != failed_label
            and cand not in self.attempted_labels
            and cand not in queued
            and self._is_valid_target(cand)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cand: (round(float(np.linalg.norm(self.layout[cand] - self.layout[failed_label])), 6), cand),
        )

    def _next_z(self):
        if self.results:
            return self.results[-1]['z_peak']
        if self.initial_z is not None:
            return self.initial_z
        return self.actual[2]

    def _build_skip_summary(self):
        primary_failures = [fp for fp in self.failed_points if fp.get('retest_of') is None]
        items = []
        skipped_points = []
        retested_points = []
        for fp in primary_failures:
            retest_lbl = fp.get('retest_label')
            retest_status = fp.get('retest_status')
            skipped = retest_status in ('failed', 'unavailable', None)
            entry = {
                'label': fp['label'],
                'error_code': fp.get('error_code'),
                'reason': fp.get('reason'),
                'retest_label': retest_lbl,
                'retest_status': retest_status,
                'retest_reason': fp.get('retest_reason'),
                'skipped': skipped,
            }
            items.append(entry)
            if retest_lbl and retest_status == 'success':
                retested_points.append(f"{fp['label']}({fp.get('reason') or fp.get('error_code')})→补测{retest_lbl}成功")
            elif retest_lbl and retest_status == 'failed':
                skipped_points.append(f"{fp['label']}({fp.get('reason') or fp.get('error_code')})→补测{retest_lbl}失败")
            elif retest_status == 'pending':
                retested_points.append(f"{fp['label']}→补测{retest_lbl}中")
            else:
                skipped_points.append(f"{fp['label']}({fp.get('reason') or fp.get('error_code')})→无可用补测点")
        parts = []
        if retested_points:
            parts.append('补测：' + '；'.join(retested_points))
        if skipped_points:
            parts.append('跳过：' + '；'.join(skipped_points))
        text = ' | '.join(parts) if parts else '跳过/补测：无失败点'
        return {
            'failed_count': len(self.failed_points),
            'primary_failed_count': len(primary_failures),
            'retested_ok_count': sum(1 for item in items if item['retest_status'] == 'success'),
            'skipped_count': sum(1 for item in items if item['skipped']),
            'items': items,
            'text': text,
        }

    def advance(self, frames, metadata=None):
        try:
            if self.phase == 'center':
                exposure_us = metadata.get('exposure_us') if isinstance(metadata, dict) else None
                return self.center(np.median(np.stack(frames), axis=0).astype(np.uint8), exposure_us=exposure_us)
            if self.phase == 'peak':
                return self.point_finished(frames[len(frames) // 2])
            return self.scan(frames)
        except GuardError as exc:
            if exc.code in SKIPPABLE_POINT_CODES:
                if exc.code == 'DEPTH-TARGET' and self.ref_xy is None:
                    raise
                return self._handle_point_failure(exc)
            raise

    def center(self, frame, exposure_us=None):
        if self.ref_xy is None:
            if self.configured_target in ('auto', 'film'):
                film_targets = self._detect_film_targets(frame)
                if film_targets:
                    self.target_type = 'film'
                    self.cal = self.film_cal
                    return self._center_film(frame, film_targets, exposure_us)
                if self.configured_target == 'film':
                    raise GuardError('DEPTH-TARGET', '未识别到菲林圆环和 ArUco 靶点', self.where,
                                     '检查初始位置、菲林方向、照明及相机焦距。')
            self._ensure_charuco()
            pts, ids = detect_charuco_corners(frame, self._charuco_detector)
            if len(ids) < 6:
                if self.configured_target == 'auto':
                    raise GuardError('DEPTH-TARGET', '未识别到菲林靶点或标定板', self.where,
                                     '检查初始位置、菲林或标定板方向、照明及相机焦距。')
                raise GuardError('DEPTH-TARGET', f'未识别到足够 ChArUco 标定板角点（检出 {len(ids)} 个，需 ≥ 6 个）',
                                 self.where, '检查初始位置、标定板方向、照明及相机焦距。')
            self.target_type = 'charuco'
            self.cal = self.board_cal
            return self._init_charuco_and_center(frame, pts, ids, exposure_us)

        if self.target_type == 'charuco':
            return self._center_charuco_point(frame, exposure_us)
        targets = self._detect_film_targets(frame)
        if not targets:
            raise GuardError('DEPTH-TARGET', '未识别到菲林圆环和 ArUco 靶点', self.where,
                             '检查初始位置、菲林方向、照明及相机焦距。')
        return self._center_film(frame, targets, exposure_us)

    def _center_film(self, frame, targets, exposure_us):
        h, w = frame.shape[:2]
        uv = ((w - 1) / 2, (h - 1) / 2)
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
        self.ref_label = target.label
        self.ref_xy = np.array(self.point_xy, dtype=float)
        if self.initial_z is None:
            self.initial_z = z
        if self.label != self.plan[self.point_index]:
            return self.locate(self.plan[self.point_index], z)
        self.attempted_labels.add(self.label)
        if math.dist(target.ring_uv, uv) > self.cfg['center_tol_px']:
            if self.center_round >= self.cfg['center_max_iter']:
                raise GuardError('DEPTH-CENTER', '达到居中轮数上限，圆环仍偏离画面中心',
                                 self.where, '检查靶点、标定投影和 PLC 到位。')
            self.center_round += 1
            return self.move((x + self.point_xy[0] - center_xy[0], y + self.point_xy[1] - center_xy[1], z))
        return self._finish_center_roi(frame, target.ring_uv, w, h, x, y, z, exposure_us)

    def _select_in_travel_candidate(self, ref_label, reserved):
        if ref_label not in self.layout:
            return None
        candidates = [
            cand for cand in self.layout
            if cand != ref_label
            and cand not in self.attempted_labels
            and cand not in reserved
            and self._is_valid_target(cand)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cand: (round(float(np.linalg.norm(self.layout[cand] - self.layout[ref_label])), 6), cand),
        )

    def _init_charuco_and_center(self, frame, pts, ids, exposure_us):
        h, w = frame.shape[:2]
        uv = ((w - 1) / 2, (h - 1) / 2)
        module_pts = np.array([pixel_to_module(u, v, self.actual, self.board_cal)[:2] for u, v in pts], dtype=float)
        board_pts = self._charuco_board_points[ids, :2]
        try:
            transform, fit_info = fit_board_to_module(board_pts, module_pts)
        except ValueError as exc:
            raise GuardError('DEPTH-TARGET', f'ChArUco 标定板二维位姿拟合失败：{exc}',
                             self.where, '检查标定板是否在视野内且角点分布非共线。') from exc
        board_center = (
            0.5 * float(self.camcal_cfg['cols']) * float(self.camcal_cfg['square_mm']),
            0.5 * float(self.camcal_cfg['rows']) * float(self.camcal_cfg['square_mm']),
        )
        outer_quad = charuco_board_outer_quad(self.camcal_cfg, transform)
        keepout_mm = float(self.cfg.get('board_keepout_mm', 5.0))
        safe_quad = charuco_safe_quad(outer_quad, keepout_mm)
        self.board_outer_quad = outer_quad
        fit_info['center_xy_mm'] = [float(v) for v in transform(board_center)]
        fit_info['board_keepout_mm'] = keepout_mm
        fit_info['outer_corners_mm'] = [[float(c[0]), float(c[1])] for c in outer_quad]
        fit_info['safe_corners_mm'] = [[float(c[0]), float(c[1])] for c in safe_quad] if safe_quad is not None else []
        self.board_fit = fit_info

        self.layout, primary, extra, self.point_positions = charuco_layout(
            self.camcal_cfg, self._charuco_board_points, transform,
        )
        center_xy = pixel_to_module(*uv, self.actual, self.board_cal)[:2]
        x, y, z = self.actual
        self.center_offset = np.array(center_xy) - (x, y)
        self.ref_label = primary[0]
        self.ref_xy = np.array(self.layout[self.ref_label], dtype=float)
        if self.initial_z is None:
            self.initial_z = z

        raw_corners = set(primary[:4])
        # 若有计划点已在行程与安全区内且靠近当前画面中心，优先从该点开始测
        in_travel_primary = [lbl for lbl in primary if self._is_valid_target(lbl)]
        ordered_primary = list(primary)
        if in_travel_primary:
            nearest_start = min(
                in_travel_primary,
                key=lambda lbl: (round(float(np.linalg.norm(self.layout[lbl] - center_xy[:2])), 6), lbl),
            )
            ordered_primary.remove(nearest_start)
            ordered_primary.insert(0, nearest_start)

        planned = []
        reserved = set(ordered_primary)
        corner_labels = set()
        self.plan_slots = {}
        self.slot_outcomes = {}
        self.retest_parent = {}
        remaining_extra = list(extra)

        for slot, lbl in enumerate(ordered_primary):
            cam_xy = self._target_camera_xy(lbl)
            in_travel = self._in_xy_travel(cam_xy)
            in_safe = self._in_board_safe_zone(cam_xy)
            if in_travel and in_safe:
                planned.append(lbl)
                self.plan_slots[lbl] = slot
                if lbl in raw_corners:
                    corner_labels.add(lbl)
                continue
            self.attempted_labels.add(lbl)
            backup = self._select_in_travel_candidate(lbl, reserved - {lbl})
            if backup is not None:
                if backup in remaining_extra:
                    remaining_extra.remove(backup)
                reserved.add(backup)
                planned.append(backup)
                self.plan_slots[backup] = slot
                self.retest_parent[backup] = lbl
                if lbl in raw_corners:
                    corner_labels.add(backup)
                if backup not in self.point_positions and lbl in self.point_positions:
                    self.point_positions[backup] = f"{self.point_positions[lbl]}(补)"
            else:
                self.slot_outcomes[slot] = 'failed'

            err_code = 'DEPTH-SAFE-ZONE' if not in_safe else 'POSITION-TRAVEL'
            err_reason = '测点超出标定板安全区' if not in_safe else '超出下针深度标定行程（depth_calibration.travel）'
            self.failed_points.append({
                'label': lbl,
                'position_name': self.point_positions.get(lbl),
                'xy': [float(v) for v in self.layout[lbl]],
                'status': 'failed',
                'error_code': err_code,
                'reason': err_reason,
                'where': 'ChArUco 测点规划',
                'phase': 'plan',
                'is_corner': lbl in raw_corners,
                'retest_of': None,
                'retest_label': backup,
                'retest_status': 'pending' if backup is not None else 'unavailable',
                'exposure_us': None, 'exposure_mode': None,
                'roi_mean': None, 'roi_p99': None,
                'fits': [], 'coarse_fit': None, 'coarse_bounds': None, 'fine_plan': None,
                'arrival': None,
            })

        final_extra = []
        for e_lbl in remaining_extra:
            if e_lbl not in reserved and self._is_valid_target(e_lbl):
                final_extra.append(e_lbl)
                reserved.add(e_lbl)
            else:
                rep = self._select_in_travel_candidate(e_lbl, reserved)
                if rep is not None:
                    final_extra.append(rep)
                    reserved.add(rep)
                    if rep not in self.point_positions and e_lbl in self.point_positions:
                        self.point_positions[rep] = f"{self.point_positions[e_lbl]}(补)"

        self.plan = planned
        self.extra = final_extra
        self.corner_labels = corner_labels
        self.skip_summary = self._build_skip_summary()
        self._check_early_fatal_rules()
        if len(self.plan) < self.cfg['min_valid_points']:
            raise GuardError(
                'DEPTH-INSUFFICIENT-POINTS',
                f'行程与安全区内可用 ChArUco 测点数 {len(self.plan)} < min_valid_points({self.cfg["min_valid_points"]})',
                'ChArUco 测点规划',
                '请将标定板移入 depth_calibration.travel 行程范围内或调整 board_keepout_mm 后重试。',
            )

        first_lbl = self.plan[0]
        first_id = charuco_corner_id(first_lbl)
        matches = np.flatnonzero(ids == first_id) if first_id is not None else np.empty(0, int)
        if len(matches) > 0:
            corner_uv = tuple(map(float, pts[int(matches[0])]))
            if math.dist(corner_uv, uv) <= self.cfg['center_tol_px']:
                self.label = first_lbl
                self.current_retest_of = self.retest_parent.get(self.label)
                self.attempted_labels.add(self.label)
                for item in self.point_arrivals:
                    if item['point_label'] is None:
                        item['point_label'] = self.label
                self.arrival_summary = self._build_arrival_summary()
                self.point_xy = pixel_to_module(*corner_uv, self.actual, self.board_cal)[:2]
                return self._finish_center_roi(frame, corner_uv, w, h, x, y, z, exposure_us)
        return self.locate(first_lbl, z)

    def _center_charuco_point(self, frame, exposure_us):
        h, w = frame.shape[:2]
        uv = ((w - 1) / 2, (h - 1) / 2)
        self._ensure_charuco()
        pts, ids = detect_charuco_corners(frame, self._charuco_detector)
        target_id = charuco_corner_id(self.label)
        matches = np.flatnonzero(ids == target_id) if target_id is not None else np.empty(0, int)
        if len(ids) < 6 or len(matches) == 0:
            raise GuardError(
                'DEPTH-TARGET',
                f'画面中未见目标角点/标定板（目标 {self.label}，检出 {len(ids)} 个角点，需 ≥ 6 个）',
                self.where,
                '检查该角点附近标定板表面、照明、行程及相机焦距。',
            )
        corner_uv = tuple(map(float, pts[int(matches[0])]))
        self.point_xy = pixel_to_module(*corner_uv, self.actual, self.board_cal)[:2]
        center_xy = pixel_to_module(*uv, self.actual, self.board_cal)[:2]
        x, y, z = self.actual
        self.center_offset = np.array(center_xy) - (x, y)
        self.attempted_labels.add(self.label)
        if math.dist(corner_uv, uv) > self.cfg['center_tol_px']:
            if self.center_round >= self.cfg['center_max_iter']:
                raise GuardError('DEPTH-CENTER', f'达到居中轮数上限，角点 {self.label} 仍偏离画面中心',
                                 self.where, '检查靶点、标定投影和 PLC 到位。')
            next_xy = (x + self.point_xy[0] - center_xy[0], y + self.point_xy[1] - center_xy[1])
            px_dx = math.dist(center_xy, pixel_to_module(uv[0] + 1.0, uv[1], self.actual, self.board_cal)[:2])
            px_dy = math.dist(center_xy, pixel_to_module(uv[0], uv[1] + 1.0, self.actual, self.board_cal)[:2])
            px_scale_mm = 0.5 * (px_dx + px_dy)
            center_tol_mm = (
                float(self.cfg['center_tol_px']) * px_scale_mm
                if math.isfinite(px_scale_mm) and px_scale_mm > 0.0 else 0.0
            )
            # 居中迭代阶段允许在严格安全区边距外额外放宽居中容差（max(0.05, center_tol_px 对应 mm × 1.5)）：
            # 现场居中允许偏差 center_tol_px（约 0.37mm）叠加板位姿拟合误差，可能使实测角点相对理论位置略向板外偏移；
            # 若仍按严格 0.05mm 判死会误杀正常居中微调，放宽后可容忍合理居中残差同时阻止明显出界；
            # 移到计划测点的下发判定（locate）与规划阶段仍使用严格边距 tol_mm=0.05。
            center_safe_tol = max(0.05, center_tol_mm * 1.5)
            if not self._in_board_safe_zone(next_xy, tol_mm=center_safe_tol):
                raise GuardError(
                    'DEPTH-SAFE-ZONE',
                    f'测点超出标定板安全区（{self.label} 居中目标）',
                    self.where,
                    '检查标定板位置、角点识别结果与 board_keepout_mm 安全区配置。',
                )
            self.center_round += 1
            return self.move((*next_xy, z))
        return self._finish_center_roi(frame, corner_uv, w, h, x, y, z, exposure_us)

    def _finish_center_roi(self, frame, center_uv, w, h, x, y, z, exposure_us):
        u, v = center_uv
        dx = math.dist(self.point_xy, pixel_to_module(u + 1, v, self.actual, self.cal)[:2])
        dy = math.dist(self.point_xy, pixel_to_module(u, v + 1, self.actual, self.cal)[:2])
        size = max(3, round(self.cfg['roi_mm'] / ((dx + dy) / 2)))
        left, top = round(u - (size - 1) / 2), round(v - (size - 1) / 2)
        if left < 0 or top < 0 or left + size > w or top + size > h:
            raise GuardError('DEPTH-ROI', f'固定 ROI {size} px 超出画面', self.where,
                             '减小 roi_mm，或检查居中和像素比例尺。')
        self.roi = np.s_[top:top + size, left:left + size]
        center_path = self.record.image(f'{self.label}_center', frame)
        self.last_center_image = str(center_path) if center_path is not None else None
        crop = frame[self.roi]
        self.record.roi(crop)
        self.camera_xy = (x, y)
        return self._prepare_exposure(crop, z, exposure_us)

    def _finish_exposure(self, crop, exposure_us, stats, z, warning=None):
        self.point_exposure_us = float(exposure_us) if exposure_us is not None else None
        self.exposure_us = self.point_exposure_us
        self.point_roi_mean = float(stats['roi_mean'])
        self.point_roi_p99 = float(stats['roi_p99'])
        self.point_roi_sat_frac = float(stats['saturated_fraction'])
        self.point_exposure_warning = warning
        self.live_roi_mean = self.point_roi_mean
        self.live_roi_p99 = self.point_roi_p99
        self.exposure_locked = True
        self._ae = None
        if warning:
            self.record.events.emit(
                'depthcal_exposure_warning',
                label=self.label,
                what=warning,
                where=self.where,
                hint='可切换手动曝光或检查光源与光圈。',
                exposure_us=self.point_exposure_us,
                roi_mean=self.point_roi_mean,
                roi_p99=self.point_roi_p99,
                saturated_fraction=self.point_roi_sat_frac,
            )
        return self._begin_coarse_up(crop, z, stats=stats)

    def _prepare_exposure(self, crop, z, exposure_us):
        mode = self.desired_exposure_mode
        manual_us = self.desired_manual_exposure_us
        self.point_exposure_mode = mode
        stats = roi_brightness(crop)
        self.live_roi_mean = stats['roi_mean']
        self.live_roi_p99 = stats['roi_p99']

        if mode == 'manual':
            if manual_us is not None:
                target_us = min(max(float(manual_us), float(self.cfg['exposure_min_us'])),
                                float(self.cfg['exposure_max_us']))
                self.phase = 'exposure'
                self._ae = {'mode': 'manual', 'target_us': target_us, 'iter': 0, 'candidates': [], 'z': z}
                return target_us
            return self._finish_exposure(
                crop, exposure_us, stats, z,
                warning='手动曝光模式未指定数值，沿用当前相机曝光' if exposure_us is not None else None,
            )

        if exposure_us is None or not math.isfinite(exposure_us) or exposure_us <= 0:
            return self._finish_exposure(crop, None, stats, z)

        cur_exp = float(exposure_us)
        next_us, warn = next_auto_exposure(cur_exp, stats, self.cfg)
        if next_us is None:
            return self._finish_exposure(crop, cur_exp, stats, z, warning=warn)
        self.phase = 'exposure'
        self._ae = {
            'mode': 'auto',
            'target_us': next_us,
            'iter': 1,
            'z': z,
            'candidates': [{'exposure_us': cur_exp, **stats}],
            'finalizing': False,
        }
        return next_us

    def _finalize_best_auto(self, ae, cur_exp, crop, stats, base_warn):
        target = float(self.cfg['exposure_target_p99'])
        best = min(
            ae['candidates'],
            key=lambda c: (c['saturated_fraction'] > EXPOSURE_MAX_SATURATED_FRACTION, abs(c['roi_p99'] - target), c['exposure_us']),
        )
        best_exp = float(best['exposure_us'])
        if abs(best_exp - cur_exp) > max(0.5, cur_exp * 1e-3):
            ae['finalizing'] = True
            ae['warning'] = f"{base_warn}，改用最接近值 {best_exp:.0f} µs（ROI p99={best['roi_p99']:.1f}）"
            ae['target_us'] = best_exp
            return 'set_exposure', best_exp
        warn = f"{base_warn}（曝光 {cur_exp:.0f} µs，ROI p99={stats['roi_p99']:.1f}，饱和={stats['saturated_fraction']*100:.2f}%）"
        return 'done', self._finish_exposure(crop, cur_exp, stats, ae['z'], warning=warn)

    def step_exposure(self, frame, exposure_us):
        """处理自动/手动曝光后的一帧采样；返回 ('set_exposure', us) 或 ('done', coords)。"""
        try:
            crop = frame[self.roi]
            self.record.roi(crop)
            stats = roi_brightness(crop)
            self.live_roi_mean = stats['roi_mean']
            self.live_roi_p99 = stats['roi_p99']
            ae = self._ae
            cur_exp = float(exposure_us) if exposure_us is not None else float(ae['target_us'])
            if ae['mode'] == 'manual':
                warn = None
                if stats['saturated_fraction'] > EXPOSURE_MAX_SATURATED_FRACTION:
                    warn = f"手动曝光 {cur_exp:.0f} µs 下 ROI 饱和像素比例 {stats['saturated_fraction']*100:.2f}% > 0.1%"
                return 'done', self._finish_exposure(crop, cur_exp, stats, ae['z'], warning=warn)
            if ae.get('finalizing'):
                return 'done', self._finish_exposure(crop, cur_exp, stats, ae['z'], warning=ae.get('warning'))

            ae['candidates'].append({'exposure_us': cur_exp, **stats})
            next_us, warn = next_auto_exposure(cur_exp, stats, self.cfg)
            if next_us is None:
                if warn is None:
                    return 'done', self._finish_exposure(crop, cur_exp, stats, ae['z'], warning=None)
                return self._finalize_best_auto(ae, cur_exp, crop, stats, warn)
            if ae['iter'] >= self.cfg['exposure_max_iter']:
                base_warn = f"自动曝光在 {self.cfg['exposure_max_iter']} 次调整后未完全达标"
                return self._finalize_best_auto(ae, cur_exp, crop, stats, base_warn)
            ae['iter'] += 1
            ae['target_us'] = next_us
            return 'set_exposure', next_us
        except GuardError as exc:
            if exc.code in SKIPPABLE_POINT_CODES and self.ref_xy is not None:
                return 'done', self._handle_point_failure(exc)
            raise

    def _begin_coarse_up(self, crop, z, *, stats=None):
        self._last_up_crop = crop
        bright = stats or roi_brightness(crop)
        self.coarse_up_rows = [{
            'z_actual': z, **scores(crop),
            'exposure_us': self.point_exposure_us,
            'roi_mean': bright['roi_mean'],
            'roi_p99': bright['roi_p99'],
        }]
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
                'exposure_us': seed_row.get('exposure_us', self.point_exposure_us),
                'roi_mean': seed_row.get('roi_mean', self.live_roi_mean),
                'roi_p99': seed_row.get('roi_p99', self.live_roi_p99),
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
            self.plan_slots = {label: idx for idx, label in enumerate(self.plan)}
        if self.label is not None and target.label != self.label:
            raise GuardError('DEPTH-TARGET', f'靶点从 {self.label} 变为 {target.label}',
                             self.where, '检查初始靶点和 PLC XY 到位坐标。')

    def locate(self, label, z):
        x, y = self._target_camera_xy(label)
        self.reset_point()
        self.label = label
        self.current_retest_of = self.retest_parent.get(label)
        self.attempted_labels.add(label)
        if self.target_type == 'charuco' and not self._in_board_safe_zone((x, y)):
            return self._handle_point_failure(GuardError(
                'DEPTH-SAFE-ZONE',
                f'测点超出标定板安全区（{label}）',
                self.where,
                '检查标定板位姿拟合与安全区边距 board_keepout_mm。',
            ))
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
            bright = roi_brightness(crops[midpoint])
            self.live_roi_mean = bright['roi_mean']
            self.live_roi_p99 = bright['roi_p99']
            row = {
                'z_actual': z,
                **{key: statistics.median(item[key] for item in metrics) for key in METRICS},
                'exposure_us': self.point_exposure_us,
                'roi_mean': bright['roi_mean'],
                'roi_p99': bright['roi_p99'],
            }
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
        bright = roi_brightness(crops[midpoint])
        self.live_roi_mean = bright['roi_mean']
        self.live_roi_p99 = bright['roi_p99']
        row = {'label': self.label, 'pass': 'coarse' if self.phase == 'coarse' else self.repeat,
               'direction': 'down', 'z_cmd': self.expected[2], 'z_actual': z,
               **{key: statistics.median(item[key] for item in metrics) for key in METRICS},
               'exposure_us': self.point_exposure_us,
               'roi_mean': bright['roi_mean'],
               'roi_p99': bright['roi_p99']}
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
        fit = None
        fit_error = None
        try:
            fit = fit_peak(self.scan_rows, primary, self.cfg['peak_method'], self.where, dof_ratio=dof_ratio)
        except GuardError as exc:
            if exc.code not in ('DEPTH-FIT', 'DEPTH-PEAK-BOUNDARY'):
                raise
            fit_error = exc
        comparisons = {primary: fit if fit is not None else {'failure': fit_error.what, 'error_code': fit_error.code}}
        for metric in METRICS:
            if metric != primary:
                try:
                    comparisons[metric] = fit_peak(self.scan_rows, metric, self.cfg['peak_method'], self.where, dof_ratio=dof_ratio)
                except GuardError as exc:
                    comparisons[metric] = {'failure': exc.what}
        if fit is not None:
            self.fits.append({'repeat': self.repeat, **fit, 'metrics': comparisons})
        else:
            self.fits.append({
                'repeat': self.repeat,
                'failed': True,
                'error_code': fit_error.code,
                'failure': fit_error.what,
                'metrics': comparisons,
            })
        failed_fits = [item for item in self.fits if item.get('failed')]
        ok_fits = [item for item in self.fits if not item.get('failed')]
        repeats_target = self.cfg['repeats_per_point']
        if repeats_target == 1 and failed_fits:
            raise fit_error
        if len(failed_fits) >= 2 or (self.repeat >= repeats_target and len(ok_fits) < 2 and repeats_target >= 2):
            code = failed_fits[0]['error_code'] if all(f.get('error_code') == failed_fits[0]['error_code'] for f in failed_fits) else 'DEPTH-FIT'
            reasons = '；'.join(f"#{f['repeat']}:{f['failure']}" for f in failed_fits)
            raise GuardError(code, f'{self.label} {repeats_target} 次重复中 {len(failed_fits)} 次失败（{reasons}）',
                             self.where, '检查该靶点清晰度曲线、曝光与扫描范围。')
        if self.repeat < repeats_target:
            return self.begin_fine()
        peaks = [item['z_peak'] for item in ok_fits]
        peak_range = max(peaks) - min(peaks)
        if len(failed_fits) == 1:
            if len(ok_fits) < 2 or peak_range > self.cfg['repeat_range_tol_mm']:
                raise GuardError(
                    'DEPTH-FIT',
                    f"{self.label} 第 {failed_fits[0]['repeat']} 次拟合失败，其余 {len(ok_fits)} 次峰位极差 "
                    f"{peak_range:.4f} mm > {self.cfg['repeat_range_tol_mm']:.3f} mm",
                    self.where,
                    '检查该靶点对焦重复性与异常曲线。',
                )
        insufficient = [item['repeat'] for item in ok_fits if not item.get('quality_ok', True)]
        covered_dofs = [item['dof_mm'] for item in ok_fits if item.get('dof_covered') and item.get('dof_mm') is not None]
        uncovered_dofs = [item['repeat'] for item in ok_fits if not item.get('dof_covered')]
        failed_repeat_details = [
            {'repeat': item['repeat'], 'code': item.get('error_code'), 'reason': item.get('failure')}
            for item in failed_fits
        ]
        self.result = {
            'label': self.label, 'position_name': self.point_positions.get(self.label),
            'xy': [float(v) for v in self.point_xy], 'fits': self.fits,
            'coarse_fit': self.coarse_fit, 'coarse_dof': self.coarse_dof,
            'coarse_bounds': self.coarse_bounds, 'fine_plan': self.fine_plan,
            'insufficient_repeats': insufficient,
            'failed_repeats': [item['repeat'] for item in failed_fits],
            'failed_repeat_details': failed_repeat_details,
            'retest_of': self.current_retest_of,
            'z_peak': statistics.mean(peaks), 'range': peak_range,
            'std': statistics.stdev(peaks) if len(peaks) > 1 else None,
            'dof_mm': statistics.mean(covered_dofs) if covered_dofs else None,
            'dof_range_mm': (max(covered_dofs) - min(covered_dofs)) if len(covered_dofs) >= 2 else None,
            'dof_covered_count': len(covered_dofs),
            'dof_uncovered': len(uncovered_dofs),
            'dof_uncovered_repeats': uncovered_dofs,
            'exposure_us': self.point_exposure_us,
            'exposure_mode': self.point_exposure_mode,
            'roi_mean': self.point_roi_mean,
            'roi_p99': self.point_roi_p99,
            'saturated_fraction': self.point_roi_sat_frac,
            'exposure_warning': self.point_exposure_warning,
            'arrival': self._build_point_arrival(self.point_arrivals),
        }
        self.results.append(self.result)
        self.phase = 'peak'
        return self.move((*self.camera_xy, self.result['z_peak']))

    def _check_early_fatal_rules(self, *, final=False):
        if ENABLE_FIRST_TWO_FAIL_ABORT and self.slot_outcomes.get(0) == 'failed' and self.slot_outcomes.get(1) == 'failed':
            first_two = [
                f"{fp['label']}（{fp.get('reason') or fp.get('error_code')}）"
                for fp in self.failed_points
                if fp.get('retest_of') is None and self.plan_slots.get(fp['label']) in (0, 1)
            ]
            detail = '；'.join(first_two) if first_two else '前两个计划点及其补测均失败'
            raise GuardError(
                'DEPTH-FIRST-TWO-FAILED',
                f'最先两个计划点都失败（触发致命条件 a，提前终止）：{detail}',
                '测点失败跳过规则（规则 a）',
                '请检查初始 Z0、曝光设置、照明及菲林固定位置后重试。',
            )
        if final or (0 in self.slot_outcomes and 1 in self.slot_outcomes):
            failed_corners = [
                lbl for lbl in sorted(self.corner_labels)
                if self.slot_outcomes.get(self.plan_slots.get(lbl)) == 'failed'
            ]
            if failed_corners:
                raise GuardError(
                    'DEPTH-CORNER-COVERAGE',
                    f"角点覆盖不足（触发致命条件 c）：角点 {'、'.join(failed_corners)} 及其补测点均失败",
                    '测点失败跳过规则（规则 c）',
                    '角点缺失将导致平面拟合只能外推，请检查该角点区域照明、菲林及行程范围。',
                )

    def _handle_point_failure(self, exc):
        failed_label = self.label or self.plan[self.point_index]
        self.attempted_labels.add(failed_label)
        retest_of = self.current_retest_of
        slot = self.plan_slots.get(failed_label)
        xy = self._target_module_xy(failed_label)
        fail_entry = {
            'label': failed_label,
            'position_name': self.point_positions.get(failed_label),
            'xy': [float(v) for v in xy] if xy is not None else None,
            'status': 'failed',
            'error_code': exc.code,
            'reason': exc.what,
            'where': exc.where,
            'phase': self.phase,
            'is_corner': failed_label in self.corner_labels,
            'retest_of': retest_of,
            'retest_label': None,
            'retest_status': None,
            'exposure_us': self.point_exposure_us,
            'exposure_mode': self.point_exposure_mode,
            'roi_mean': self.point_roi_mean,
            'roi_p99': self.point_roi_p99,
            'fits': list(self.fits),
            'coarse_fit': self.coarse_fit,
            'coarse_bounds': self.coarse_bounds,
            'fine_plan': self.fine_plan,
            'arrival': self._build_point_arrival(self.point_arrivals),
        }
        self.failed_points.append(fail_entry)

        if retest_of is None:
            backup = self._select_backup_target(failed_label)
            if backup is not None:
                if backup in self.extra:
                    self.extra.remove(backup)
                fail_entry['retest_label'] = backup
                fail_entry['retest_status'] = 'pending'
                self.retest_parent[backup] = failed_label
                if slot is not None:
                    self.plan_slots[backup] = slot
                self.plan.insert(self.point_index + 1, backup)
                self.skip_summary = self._build_skip_summary()
                self.point_index += 1
                return self.locate(backup, self._next_z())
            fail_entry['retest_status'] = 'unavailable'
            if slot is not None:
                self.slot_outcomes[slot] = 'failed'
        else:
            orig = next(
                (fp for fp in reversed(self.failed_points) if fp['label'] == retest_of and fp.get('retest_of') is None),
                None,
            )
            if orig is not None:
                orig['retest_status'] = 'failed'
                orig['retest_reason'] = exc.what
                orig['retest_error_code'] = exc.code
            if slot is not None:
                self.slot_outcomes[slot] = 'failed'

        self.skip_summary = self._build_skip_summary()
        self._check_early_fatal_rules()
        self.point_index += 1
        return self._advance_after_point(self._next_z())

    def _advance_after_point(self, z_next):
        if self.point_index < len(self.plan):
            return self.locate(self.plan[self.point_index], z_next)
        self._check_early_fatal_rules(final=True)
        min_valid = self.cfg['min_valid_points']
        if len(self.results) < min_valid:
            raise GuardError(
                'DEPTH-INSUFFICIENT-POINTS',
                f'最终有效测点数 {len(self.results)} < min_valid_points({min_valid})（触发致命条件 b）',
                '测点失败跳过规则（规则 b）',
                '有效测点不足，请检查失败测点原因并调整参数后重试。',
            )
        self.plane = fit_plane(
            self.results,
            surface_type=self.target_type or 'film',
            board_thickness_mm=self.board_thickness_mm,
        )
        tol = self.cfg['plane_tol_mm']
        avail_extra = [
            lbl for lbl in self.extra
            if lbl not in self.attempted_labels and self._is_valid_target(lbl)
        ]
        slots = max(0, self.cfg['max_points'] - len(self.results))
        if self.initial_plane is None:
            self.initial_plane = self.plane
            if self.plane['max_residual'] > tol:
                added = avail_extra[:slots]
                if not added:
                    raise GuardError(
                        'DEPTH-PLANE-RESIDUAL',
                        f"有效点平面最大残差 {self.plane['max_residual']:.4f} mm > plane_tol_mm({tol:.4f} mm)，且无可加测点（触发致命条件 d）",
                        '平面判定（规则 d）',
                        '料盘不平且已无可用加测靶点，请检查料盘平整度或行程配置。',
                    )
                for lbl in added:
                    self.extra.remove(lbl)
                self.plan.extend(added)
        else:
            delta = self.plane['max_residual'] - self.initial_plane['max_residual']
            self.plane['after_extra'] = ('加测后残差下降' if delta < -1e-6 else
                                         '加测后残差增大' if delta > 1e-6 else '加测后残差不变')
            if self.plane['max_residual'] > tol:
                added = avail_extra[:slots]
                if added:
                    for lbl in added:
                        self.extra.remove(lbl)
                    self.plan.extend(added)
                else:
                    raise GuardError(
                        'DEPTH-PLANE-RESIDUAL',
                        f"有效点平面最大残差 {self.plane['max_residual']:.4f} mm > plane_tol_mm({tol:.4f} mm)，且已无可加测点（max_points={self.cfg['max_points']}，触发致命条件 d）",
                        '平面判定（规则 d）',
                        '料盘平整度超差且已达到最大点数或无剩余可加测点，请检查料盘与各点曲线。',
                    )
        if self.point_index < len(self.plan):
            return self.locate(self.plan[self.point_index], z_next)
        self.feasibility = repeatability(
            self.results,
            self.cfg['repeat_range_tol_mm'],
            plane=self.plane,
            target_type=self.target_type or 'film',
        )
        self.dof_summary = dof_summary(self.results, self.cfg['dof_ratio'])
        self.skip_summary = self._build_skip_summary()
        return None

    def point_finished(self, frame):
        self.record.image(f'{self.label}_peak', frame)
        crop = frame[self.roi]
        self.record.roi(crop)
        bright = roi_brightness(crop)
        self.live_roi_mean = bright['roi_mean']
        self.live_roi_p99 = bright['roi_p99']
        self.result['peak_image_z'] = self.actual[2]
        if self.current_retest_of:
            orig = next(
                (fp for fp in reversed(self.failed_points)
                 if fp['label'] == self.current_retest_of and fp.get('retest_of') is None),
                None,
            )
            if orig is not None:
                orig['retest_status'] = 'success'
                orig['resolved'] = True
        slot = self.plan_slots.get(self.label)
        if slot is not None:
            self.slot_outcomes[slot] = 'retested_ok' if self.current_retest_of else 'ok'
        self.skip_summary = self._build_skip_summary()
        self._check_early_fatal_rules()
        self.point_index += 1
        return self._advance_after_point(self.result['z_peak'])

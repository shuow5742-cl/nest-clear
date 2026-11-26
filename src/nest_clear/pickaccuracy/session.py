"""挑取精度测量策略；设备和 PLC I/O 留在流程层。"""
from collections import defaultdict
import math
import numpy as np

from nest_clear.guard import GuardError
from nest_clear.tooloffset.session import Session as OffsetSession, fit_similarity
from nest_clear.vision.geometry import pixel_to_module
from .store import evaluate

DEFAULTS = dict(center_tol_mm=.5, infield_warn_mm=.05, const_max_dev_mm=.05,
                const_std_mm=.025, repeat_tol_mm=.03, verify_tol_mm=.05,
                outlier_mm=.1, max_points=40, calib_tool_z_mm=-10.,
                min_points_a=9, min_points_b=16, min_infield_corners=4,
                z_flat_tol_mm=.1, z_settle_samples=6, z_settle_window_s=.5,
                z_jitter_tol_mm=.02)


class Session(OffsetSession):
    def __init__(self, calibration, start, travel, config, tool, offsets, fine_half, board_thickness_mm=0., prior=None):
        super().__init__(calibration, start, travel, config, 'center', tool, board_thickness_mm)
        self.cfg = {**self.cfg, **DEFAULTS, **config}
        if self.cfg['center_tol_mm'] > fine_half / 2:
            raise ValueError('center_tol_mm 不能超过精定位框半宽的一半')
        self.offsets, self.fine_half, self.prior = offsets, fine_half, prior
        self.phase = 'overview'; self.samples = []; self.edges = []; self.skipped = []
        self.queue = []; self.candidate = None; self.sent = None; self.plan = None
        self.pending_sample = None
        self.simple = bool(tool == 2 and prior)
        self.plans = {}
        self.available = None
        self.retested = set(); self.failure = None; self.decision = {}; self.warnings = []
        self.attempt_count = 0

    def _queue(self, indices, stage):
        self.queue += [(i, stage, None) for i in indices]

    def _reachable(self, index, stage, corner=None, record=True):
        pos = self.predicted[index]
        desired = np.zeros(2) if corner is None else np.asarray(corner) * (.8 * self.fine_half)
        raw = self._pose_for(pos, (self.cal.K[0,2], self.cal.K[1,2]))
        raw = (raw[0]-desired[0], raw[1]-desired[1], raw[2])
        pose = self._clamp_pose(raw)
        uv = self._project(pos, pose)
        if not self._within(pose) or uv is None or not self._in_image(uv):
            reason = '相机不可达'
        else:
            comp = self._verification_comp(index, pos) if stage == 'verify' else (0., 0.)
            offset = self.offsets[self.tool-1]
            toolpos = (pos[0]+offset[0]+comp[0], pos[1]+offset[1]+comp[1], self.cfg['calib_tool_z_mm'])
            reason = None if self._within(toolpos) else '工具不可达'
        if reason:
            entry = {'target':f'T{index:02d}', 'stage':stage, 'reason':reason}
            if corner is not None: entry['corner'] = corner
            if record and entry not in self.skipped: self.skipped.append(entry)
            return None
        return pose, uv, not np.allclose(pose, raw, atol=1e-9, rtol=0)

    def _spread(self, indices, count, anchors=()):
        remaining = sorted(indices); selected = []; anchors = list(anchors)
        while remaining and len(selected) < count:
            if anchors or selected:
                def score(i):
                    return min(np.linalg.norm(self.predicted[i]-self.predicted[j]) for j in anchors+selected)
            else:
                def score(i):
                    p = self.predicted[i] + np.asarray(self.offsets[self.tool-1][:2])
                    return min(p[0]-self.travel[0][0], self.travel[0][1]-p[0],
                               p[1]-self.travel[1][0], self.travel[1][1]-p[1])
            chosen = max(remaining, key=score); remaining.remove(chosen); selected.append(chosen)
        return selected

    def _start_plan(self):
        self.available = [i for i in sorted(self.predicted) if self._reachable(i, 'A')]
        stage = 'simple' if self.simple else 'A'
        selected = self._spread(self.available, 5 if self.simple else 9)
        self.plans[stage] = [f'T{i:02d}' for i in selected]
        self._queue(selected, stage)

    def _plan_verify(self, count):
        fitted = {int(s['target'][1:]) for s in self._fit_samples()}
        available = []
        for i in sorted(self.predicted):
            try:
                if self._reachable(i, 'verify'): available.append(i)
            except (ValueError, RuntimeError):
                continue
        fresh = self._spread([i for i in available if i not in fitted], count, fitted)
        chosen = fresh + self._spread([i for i in available if i in fitted], count-len(fresh), fresh)
        self.plans.setdefault('verify', []).append([f'T{i:02d}' for i in chosen])
        self._queue(chosen, 'verify')

    def _move_next(self):
        if self.phase == 'failed': return 'failed'
        if self.available is None:
            self._start_plan()
        if self.queue and self.attempt_count >= self.cfg['max_points']:
            return self._fail('超过最大测点数')
        while self.queue:
            index, stage, corner = self.queue.pop(0)
            if stage == 'repeat' and not any(s['target'] == f'T{index:02d}' for s in self._fit_samples()):
                continue
            reachable = self._reachable(index, stage, corner)
            if reachable is None: continue
            pose, uv, clamped = reachable
            self.current, self.plan, self.target_pose = index, (stage, corner), pose
            self.expected_uv, self.camera_clamped = uv, clamped
            self.target_uv = None
            self.retries, self.phase = 0, 'center'
            return 'continue'
        return self._advance()

    def _verification_comp(self, index, pos):
        if self.candidate and self.candidate['type'] == 'table' and not self.simple:
            # 节点留一验证；只用其余节点预测。
            nodes = [n for n in self.candidate['nodes'] if n['target'] != f'T{index:02d}']
            return evaluate({'type':'table','nodes':nodes}, pos)
        return evaluate(self.candidate, pos)

    def image(self, targets, pose, image_size):
        self.image_size = image_size
        targets = {t.index:t for t in targets}
        if self.phase == 'overview':
            points = {i:pixel_to_module(*t.ring_uv, pose, self.cal)[:2] for i,t in targets.items() if i in self.nominal}
            if len(points) < 2: return self._fail('未找到标定卡')
            matrix, trans, scale = fit_similarity([self.nominal[i] for i in points], list(points.values()))
            if abs(scale-1) > .03: return self._fail('标定卡尺寸不符')
            self.predicted = {i:matrix @ p + trans for i,p in self.nominal.items()}
            return self._move_next()
        target = targets.get(self.current)
        if target is None: return self._fail(f'无法识别目标 T{self.current:02d}')
        if target.cross_error_mm > 2*self.cfg['ring_cross_tol_mm']: return self._fail('目标识别质量差')
        point = np.asarray(pixel_to_module(*target.ring_uv, pose, self.cal)[:2])
        expected = self.expected_uv if self.expected_uv is not None else (self.cal.K[0,2], self.cal.K[1,2])
        principal = np.asarray(pixel_to_module(*expected, pose, self.cal)[:2])
        error = point-principal
        if np.linalg.norm(error) > self.cfg['center_tol_mm']:
            raw = (pose[0]+error[0], pose[1]+error[1], self.start[2])
            corrected = self._clamp_pose(raw)
            if np.allclose(corrected, pose, atol=1e-9, rtol=0):
                self.camera_clamped = True
                if not self._in_image(target.ring_uv): return self._fail('目标超出画面中央 80% 范围')
            elif self.retries >= 2:
                return self._fail('目标对中重试超限')
            else:
                self.retries += 1; self.target_pose = corrected
                self.camera_clamped = self.camera_clamped or not np.allclose(corrected, raw, atol=1e-9, rtol=0)
                return 'continue'
        if self.camera_clamped and not self._in_image(target.ring_uv): return self._fail('目标超出画面中央 80% 范围')
        self.target_uv = list(target.ring_uv)
        self.measured[self.current] = point
        self.phase = 'tool'
        return 'tool'

    def add_xy_sample(self, actual):
        """记录本测点 XY 对准样本并进入下针 Z 测量阶段。"""
        if self.phase != 'tool' or self.sent is None:
            raise ValueError('当前不等待工具到位')
        index, (stage, corner) = self.current, self.plan
        self.attempt_count += 1
        residual = np.asarray(actual[:2]) - np.asarray(self.sent[:2])
        self.pending_sample = {
            'target': f'T{index:02d}',
            'target_xy_mm': self.measured[index].tolist(),
            'camera_xyz_mm': [float(v) for v in getattr(self, 'capture_pose', self.target_pose)],
            'sent_xyz_mm': [float(v) for v in self.sent],
            'actual_xyz_mm': [float(v) for v in actual],
            'residual_xy_mm': residual.tolist(),
            'stage': stage,
            'corner': corner,
            'outlier': False,
            'retest': stage in ('repeat', 'outlier_retest'),
            'verification': stage == 'verify',
            'camera_clamped': bool(self.camera_clamped),
            'target_uv': self.target_uv,
        }
        self.phase = 'z_step'
        return 'z_step'

    def record_z(self, z, z_jitter_mm=None):
        """记录本测点下针 Z（不加减任何量）并校验 Z 行程后推进流程。"""
        if self.phase != 'z_step' or self.pending_sample is None:
            raise ValueError('当前不等待记录下针 Z')
        z_val = float(z)
        lo, hi = self.travel[2]
        if not math.isfinite(z_val) or not lo <= z_val <= hi:
            raise GuardError(
                'POSITION-TRAVEL',
                f'下针 Z {z_val} 超出 Z 行程范围 ({lo}, {hi})',
                f'挑取精度标定 T{self.current:02d}',
                '请将 Z 轴点动回行程范围内后重新按【记录下针 Z】。',
            )
        sample = dict(self.pending_sample)
        sample['pick_z_mm'] = z_val
        if z_jitter_mm is not None:
            sample['z_jitter_mm'] = float(z_jitter_mm)
        self.pending_sample = None
        return self._commit_sample(sample)

    def _commit_sample(self, sample):
        index = int(sample['target'][1:])
        stage, corner = sample['stage'], sample['corner']
        residual = np.asarray(sample['residual_xy_mm'])
        self.samples.append(sample)
        if stage == 'infield':
            base = [s['residual_xy_mm'] for s in self.samples if s['target'] == sample['target'] and s['stage'] in ('A', 'simple', 'outlier_retest') and not s.get('discarded')]
            distance = float(np.linalg.norm(residual - np.mean(base, axis=0)))
            self.edges.append({'corner': corner, 'distance_mm': distance, 'warning': distance > self.cfg['infield_warn_mm']})
            if self.edges[-1]['warning']:
                self.warnings.append('框内误差不均，相机标定需复核')
        elif stage == 'repeat':
            first = next(s for s in self.samples if s['target'] == sample['target'] and s['stage'] in ('A', 'B', 'simple', 'outlier_retest') and not s.get('discarded'))
            if np.linalg.norm(residual - first['residual_xy_mm']) > self.cfg['repeat_tol_mm']:
                return self._fail('误差不可重复：检查机械间隙/标定卡是否移动/人工对准')
        elif stage == 'outlier_retest':
            first = next(s for s in self.samples if s['target'] == sample['target'] and s.get('outlier') and not s.get('resolved'))
            first['resolved'] = True
            if np.linalg.norm(residual - first['residual_xy_mm']) > self.cfg['repeat_tol_mm']:
                first['discarded'] = True
        elif stage in ('A', 'B'):
            baseline = [s['residual_xy_mm'] for s in self.samples[:-1] if s['stage'] in ('A', 'B', 'simple') and not s.get('discarded')]
            if len(baseline) >= 3 and np.linalg.norm(residual - np.mean(baseline, axis=0)) > self.cfg['outlier_mm'] and index not in self.retested:
                sample['outlier'] = True
                self.retested.add(index)
                self.queue.insert(0, (index, 'outlier_retest', None))
        return self._move_next()

    def add_sample(self, actual):
        self.add_xy_sample(actual)
        return self.record_z(actual[2])

    def discard(self):
        if self.phase not in ('tool', 'z_step'):
            raise ValueError('当前不在可作废的测点阶段')
        if self.phase == 'tool':
            self.attempt_count += 1
        self.pending_sample = None
        self.samples.append({'target':f'T{self.current:02d}', 'target_xy_mm':self.measured[self.current].tolist(),
                             'stage':'discard', 'discarded':True, 'reason':'本点作废重测'})
        self.queue.insert(0,(self.current,*self.plan))
        self.skipped.append({'target':f'T{self.current:02d}', 'stage':self.plan[0], 'reason':'本点作废重测'})
        return self._move_next()

    def _fit_samples(self):
        return [s for s in self.samples if s['stage'] in ('A','B','simple','repeat','outlier_retest') and not s.get('discarded')]

    def _constant(self, samples):
        values = np.asarray([s['residual_xy_mm'] for s in samples]); mean = values.mean(axis=0)
        d = np.linalg.norm(values-mean,axis=1)
        maximum, std = float(d.max()), float(d.std())
        self.decision = {'max_dev_mm':maximum, 'std_mm':std}
        return mean, maximum <= self.cfg['const_max_dev_mm'] and std <= self.cfg['const_std_mm']

    def _table(self):
        grouped = defaultdict(list)
        for s in self._fit_samples(): grouped[s['target']].append(s)
        if any(f'T{i:02d}' not in grouped for i in self.b_nodes): raise ValueError('补偿表测点不足或不可达')
        nodes=[]
        for i in self.b_nodes:
            rows=grouped[f'T{i:02d}']; xy=np.mean([s['target_xy_mm'] for s in rows],axis=0); r=np.mean([s['residual_xy_mm'] for s in rows],axis=0)
            nodes.append(dict(target=f'T{i:02d}', x=float(xy[0]), y=float(xy[1]), dx=float(r[0]), dy=float(r[1]), n=len(rows)))
        points = np.asarray([[n['x'], n['y']] for n in nodes])
        if len(points) < 3 or np.linalg.matrix_rank(points - points[0], tol=1e-8) < 2:
            raise ValueError('补偿表节点共线，无法生成补偿表')
        return {'type':'table','nodes':nodes}

    def _advance(self):
        # 队列测完后再次按全体均值检查，覆盖首个测点及随后均值变化的情况。
        if self.candidate is None:
            rows=[s for s in self._fit_samples() if s['stage'] in ('A','B','simple')]
            if len(rows)>=3:
                values=np.asarray([np.asarray(s['residual_xy_mm']) - (np.asarray(evaluate(self.prior,s['target_xy_mm'])) if self.simple else 0) for s in rows])
                center=values.mean(axis=0)
                for sample,value in zip(rows,values):
                    index=int(sample['target'][1:])
                    if np.linalg.norm(value-center)>self.cfg['outlier_mm'] and index not in self.retested:
                        sample['outlier']=True; self.retested.add(index); self.queue.append((index,'outlier_retest',None))
                if self.queue: return self._move_next()
        stages = {s['stage'] for s in self.samples}
        if self.simple and self.candidate and getattr(self, 'simple_done', False):
            verification=[s for s in self.samples if s['stage']=='verify' and not s.get('superseded')]
            if not verification: return self._fail('验证点不足或不可达')
            if all(np.linalg.norm(s['residual_xy_mm']) <= self.cfg['verify_tol_mm'] for s in verification):
                self.phase='confirm'; return 'confirm'
            for s in verification: s['superseded']=True
            self.simple=False; self.candidate=None
            self._resume_full(); return self._move_next()
        if self.simple and 'simple_done' not in stages and not getattr(self,'simple_done',False):
            rows=[s for s in self._fit_samples() if s['stage'] in ('simple','outlier_retest')]
            if len({s['target'] for s in rows}) < self.cfg['min_points_a']:
                self.simple=False; self._resume_full(); return self._move_next()
            diffs=np.asarray([np.asarray(s['residual_xy_mm'])-evaluate(self.prior,s['target_xy_mm']) for s in rows]); mean=diffs.mean(axis=0); d=np.linalg.norm(diffs-mean,axis=1)
            self.simple_done=True
            if d.max() <= self.cfg['const_max_dev_mm'] and d.std() <= self.cfg['const_std_mm']:
                if self.prior['type']=='constant': self.candidate={'type':'constant','constant_xy':(np.asarray(self.prior['constant_xy'])+mean).tolist()}
                else: self.candidate={'type':'table','nodes':[dict(n,dx=n['dx']+float(mean[0]),dy=n['dy']+float(mean[1])) for n in self.prior['nodes']]}
                self._plan_verify(1); return self._move_next()
            self.simple=False; self._resume_full(); return self._move_next()
        if not getattr(self,'infield_done',False):
            rows = [s for s in self._fit_samples() if s['stage'] in ('A','simple','outlier_retest')]
            indices = sorted({int(s['target'][1:]) for s in rows})
            if len(indices) < self.cfg['min_points_a']:
                skipped = [s for s in self.skipped if s['stage']=='A' and s['reason'] in ('相机不可达','工具不可达')]
                a = sum(s['reason']=='相机不可达' for s in skipped); b = sum(s['reason']=='工具不可达' for s in skipped)
                return self._fail(f'阶段 A 可用目标不足：已测 {len(indices)} 个，需 {self.cfg["min_points_a"]} 个；32 个目标中 {a+b} 个因相机/工具超出行程被跳过（相机 {a} 个、工具 {b} 个）')
            center = np.mean([self.predicted[i] for i in indices], axis=0)
            index = min(indices, key=lambda i: np.linalg.norm(self.predicted[i]-center))
            self.infield_done=True
            corners = [c for c in ((-1,-1),(-1,1),(1,-1),(1,1)) if self._reachable(index,'infield',c)]
            self.plans['infield'] = {'target':f'T{index:02d}', 'corners':corners}
            self.queue += [(index,'infield',c) for c in corners]; return self._move_next()
        if not getattr(self,'classified',False):
            if len(self.edges) < self.cfg['min_infield_corners']: return self._fail('视野内检查点不足或不可达')
            mean, good=self._constant(self._fit_samples()); self.classified=True
            if good:
                self.candidate={'type':'constant','constant_xy':mean.tolist()}; self._plan_verify(2)
            else: self._begin_b()
            return self._move_next()
        if getattr(self,'in_b',False) and not getattr(self,'table_ready',False):
            if len({s['target'] for s in self.samples if s['stage']=='repeat'}) < 3:
                return self._fail('重复性检查点不足或不可达')
            try: self.candidate=self._table()
            except ValueError as exc: return self._fail(str(exc))
            self.table_ready=True; self._plan_verify(2); return self._move_next()
        verification=[s for s in self.samples if s['stage']=='verify' and not s.get('superseded')]
        expected=1 if self.simple else 2
        if len(verification)<expected: return self._fail('验证点不足或不可达')
        if any(np.linalg.norm(s['residual_xy_mm']) > self.cfg['verify_tol_mm'] for s in verification):
            if self.candidate['type']=='constant' and not getattr(self,'in_b',False):
                for s in verification: s['superseded']=True
                self.simple=False; self._begin_b(); return self._move_next()
            return self._fail('补偿后仍超差')
        self.phase='confirm'; return 'confirm'

    def _resume_full(self):
        existing = {int(s['target'][1:]) for s in self._fit_samples()}
        selected = list(existing) + self._spread([i for i in self.available if i not in existing], max(0,9-len(existing)), existing)
        self.plans['A'] = [f'T{i:02d}' for i in selected]
        self._queue([i for i in selected if i not in existing], 'A')

    def _begin_b(self):
        self.in_b=True; self.candidate=None
        existing={int(s['target'][1:]) for s in self._fit_samples()}
        # Preserve A measurements as nodes, then fill remaining slots by coverage.
        included = sorted(existing.intersection(self.available))
        self.b_nodes = included + self._spread([i for i in self.available if i not in included], 16-len(included), included)
        self.plans['B'] = [f'T{i:02d}' for i in self.b_nodes]
        if len(self.b_nodes) < self.cfg['min_points_b']:
            self.failure = '补偿表测点不足或不可达'; self.phase='failed'; return
        self._queue([i for i in self.b_nodes if i not in existing],'B')
        repeat = self._spread(self.b_nodes, 3)
        self.plans['repeat'] = [f'T{i:02d}' for i in repeat]
        self._queue(repeat,'repeat')

    def z_summary(self):
        """汇总全部非作废测点的下针 Z，按最大差阈值决定全盘最大值或位置 Z 表。"""
        z_samples = [s for s in self.samples if not s.get('discarded') and s.get('pick_z_mm') is not None]
        if not z_samples:
            return {}
        grouped = defaultdict(list)
        for s in z_samples:
            grouped[s['target']].append(s)
        nodes = []
        for target_name in sorted(grouped):
            rows = grouped[target_name]
            xy = np.mean([s['target_xy_mm'] for s in rows], axis=0)
            z = float(np.mean([s['pick_z_mm'] for s in rows]))
            nodes.append({'target': target_name, 'x': float(xy[0]), 'y': float(xy[1]), 'z': z, 'n': len(rows)})
        zs = [n['z'] for n in nodes]
        z_min, z_max = float(min(zs)), float(max(zs))
        z_range = float(z_max - z_min)
        threshold = float(self.cfg['z_flat_tol_mm'])
        z_decision = {
            'z_type': 'constant' if z_range <= threshold else 'table',
            'min_z_mm': z_min,
            'max_z_mm': z_max,
            'range_mm': z_range,
            'threshold_mm': threshold,
            'point_count': len(nodes),
        }
        if z_range <= threshold:
            return {'z_type': 'constant', 'z_mm': z_max, 'z_decision': z_decision}
        return {'z_type': 'table', 'z_nodes': nodes, 'z_decision': z_decision}

    def result(self):
        return dict(self.candidate, **self.z_summary(), samples=self.samples, infield=self.edges, warnings=self.warnings,
                    skipped=self.skipped, decision=self.decision, plan=self.plans,
                    verification=[s for s in self.samples if s['stage']=='verify' and not s.get('superseded')])

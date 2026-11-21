"""挑取精度测量策略；设备和 PLC I/O 留在流程层。"""
from collections import defaultdict
import math
import numpy as np

from nest_clear.tooloffset.session import Session as OffsetSession, fit_similarity
from nest_clear.vision.geometry import pixel_to_module
from .store import evaluate

DEFAULTS = dict(center_tol_mm=.5, infield_warn_mm=.05, const_max_dev_mm=.05,
                const_std_mm=.025, repeat_tol_mm=.03, verify_tol_mm=.05,
                outlier_mm=.1, max_points=40, calib_tool_z_mm=-10.)
A = (2, 3, 5, 10, 12, 21, 26, 28, 29)
B = (2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29)
SIMPLE = (2, 5, 26, 29, 12)


class Session(OffsetSession):
    def __init__(self, calibration, start, travel, config, tool, offsets, fine_half, board_thickness_mm=0., prior=None):
        super().__init__(calibration, start, travel, config, 'center', tool, board_thickness_mm)
        self.cfg = {**self.cfg, **DEFAULTS, **config}
        if self.cfg['center_tol_mm'] > fine_half / 2:
            raise ValueError('center_tol_mm 不能超过精定位框半宽的一半')
        self.offsets, self.fine_half, self.prior = offsets, fine_half, prior
        self.phase = 'overview'; self.samples = []; self.edges = []; self.skipped = []
        self.queue = []; self.candidate = None; self.sent = None; self.plan = None
        self.simple = bool(tool == 2 and prior)
        self.queue = [(i, 'simple' if self.simple else 'A', None) for i in (SIMPLE if self.simple else A)]
        self.retested = set(); self.failure = None; self.decision = {}; self.warnings = []
        self.attempt_count = 0

    def _queue(self, indices, stage):
        self.queue += [(i, stage, None) for i in indices]

    def _move_next(self):
        if self.queue and self.attempt_count >= self.cfg['max_points']:
            return self._fail('超过最大测点数')
        while self.queue:
            index, stage, corner = self.queue.pop(0)
            pos = self.predicted[index]
            desired = np.zeros(2) if corner is None else np.array(corner) * (.8 * self.fine_half)
            pose = self._pose_for(pos, (self.cal.K[0,2], self.cal.K[1,2]))
            pose = (pose[0] - desired[0], pose[1] - desired[1], pose[2])
            offset = self.offsets[self.tool-1]
            compensation = self._verification_comp(index, pos) if stage == 'verify' else (0., 0.)
            toolpos = (pos[0]+offset[0]+compensation[0], pos[1]+offset[1]+compensation[1], self.cfg['calib_tool_z_mm'])
            if not self._within(pose) or not self._within(toolpos):
                self.skipped.append({'target':f'T{index:02d}', 'stage':stage, 'reason':'相机或工具坐标超出行程'})
                continue
            self.current, self.plan, self.target_pose = index, (stage, corner), pose
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
        principal = np.asarray(pixel_to_module(self.cal.K[0,2], self.cal.K[1,2], pose, self.cal)[:2])
        corner = self.plan[1]
        desired = np.zeros(2) if corner is None else np.asarray(corner)*(.8*self.fine_half)
        error = point-principal-desired
        if np.linalg.norm(error) > self.cfg['center_tol_mm']:
            if self.retries >= 2: return self._fail('目标对中重试超限')
            self.retries += 1
            self.target_pose = (pose[0]+error[0], pose[1]+error[1], self.start[2])
            if not self._within(self.target_pose): return self._fail('修正后的相机坐标超出行程')
            return 'continue'
        self.measured[self.current] = point
        self.phase = 'tool'
        return 'tool'

    def add_sample(self, actual):
        if self.phase != 'tool' or self.sent is None: raise ValueError('当前不等待工具到位')
        index, (stage, corner) = self.current, self.plan
        self.attempt_count += 1
        residual = np.asarray(actual[:2])-np.asarray(self.sent[:2])
        sample = {'target':f'T{index:02d}', 'target_xy_mm':self.measured[index].tolist(),
                  'camera_xyz_mm':[float(v) for v in getattr(self, 'capture_pose', self.target_pose)], 'sent_xyz_mm':[float(v) for v in self.sent],
                  'actual_xyz_mm':[float(v) for v in actual], 'residual_xy_mm':residual.tolist(),
                  'stage':stage, 'corner':corner, 'outlier':False, 'retest':stage in ('repeat','outlier_retest'),
                  'verification':stage == 'verify'}
        self.samples.append(sample)
        if stage == 'infield':
            base = [s['residual_xy_mm'] for s in self.samples if s['target'] == sample['target'] and s['stage'] in ('A','simple','outlier_retest') and not s.get('discarded')]
            distance = float(np.linalg.norm(residual - np.mean(base,axis=0)))
            self.edges.append({'corner':corner, 'distance_mm':distance, 'warning':distance > self.cfg['infield_warn_mm']})
            if self.edges[-1]['warning']: self.warnings.append('框内误差不均，相机标定需复核')
        elif stage == 'repeat':
            first = next(s for s in self.samples if s['target']==sample['target'] and s['stage'] in ('A','B','simple','outlier_retest') and not s.get('discarded'))
            if np.linalg.norm(residual-first['residual_xy_mm']) > self.cfg['repeat_tol_mm']:
                return self._fail('误差不可重复：检查机械间隙/标定卡是否移动/人工对准')
        elif stage == 'outlier_retest':
            first = next(s for s in self.samples if s['target']==sample['target'] and s.get('outlier') and not s.get('resolved'))
            first['resolved'] = True
            if np.linalg.norm(residual-first['residual_xy_mm']) > self.cfg['repeat_tol_mm']:
                first['discarded'] = True
        elif stage in ('A','B'):
            baseline = [s['residual_xy_mm'] for s in self.samples[:-1] if s['stage'] in ('A','B','simple') and not s.get('discarded')]
            if len(baseline) >= 3 and np.linalg.norm(residual-np.mean(baseline,axis=0)) > self.cfg['outlier_mm'] and index not in self.retested:
                sample['outlier'] = True; self.retested.add(index)
                self.queue.insert(0,(index,'outlier_retest',None))
        return self._move_next()

    def discard(self):
        self.attempt_count += 1
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
        if any(f'T{i:02d}' not in grouped for i in B): raise ValueError('补偿表测点不足或不可达')
        nodes=[]
        for i in B:
            rows=grouped[f'T{i:02d}']; xy=np.mean([s['target_xy_mm'] for s in rows],axis=0); r=np.mean([s['residual_xy_mm'] for s in rows],axis=0)
            nodes.append(dict(target=f'T{i:02d}', x=float(xy[0]), y=float(xy[1]), dx=float(r[0]), dy=float(r[1]), n=len(rows)))
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
            self._queue([i for i in A if i not in SIMPLE], 'A'); return self._move_next()
        if self.simple and 'simple_done' not in stages and not getattr(self,'simple_done',False):
            rows=[s for s in self._fit_samples() if s['stage'] in ('simple','outlier_retest')]
            if len({s['target'] for s in rows}) < 5: return self._fail('简化流程测点不足')
            diffs=np.asarray([np.asarray(s['residual_xy_mm'])-evaluate(self.prior,s['target_xy_mm']) for s in rows]); mean=diffs.mean(axis=0); d=np.linalg.norm(diffs-mean,axis=1)
            self.simple_done=True
            if d.max() <= self.cfg['const_max_dev_mm'] and d.std() <= self.cfg['const_std_mm']:
                if self.prior['type']=='constant': self.candidate={'type':'constant','constant_xy':(np.asarray(self.prior['constant_xy'])+mean).tolist()}
                else: self.candidate={'type':'table','nodes':[dict(n,dx=n['dx']+float(mean[0]),dy=n['dy']+float(mean[1])) for n in self.prior['nodes']]}
                self._queue((19,), 'verify'); return self._move_next()
            self.simple=False; self._queue([i for i in A if i not in SIMPLE], 'A'); return self._move_next()
        if not getattr(self,'infield_done',False):
            if len({s['target'] for s in self._fit_samples()}) < len(A): return self._fail('阶段 A 测点不足或不可达')
            self.infield_done=True; self.queue += [(12,'infield',c) for c in ((-1,-1),(-1,1),(1,-1),(1,1))]; return self._move_next()
        if not getattr(self,'classified',False):
            if len(self.edges) != 4: return self._fail('视野内检查点不足或不可达')
            mean, good=self._constant(self._fit_samples()); self.classified=True
            if good:
                self.candidate={'type':'constant','constant_xy':mean.tolist()}; self._queue((19,11),'verify')
            else: self._begin_b()
            return self._move_next()
        if getattr(self,'in_b',False) and not getattr(self,'table_ready',False):
            if {s['target'] for s in self.samples if s['stage']=='repeat'} != {'T02','T29','T12'}:
                return self._fail('重复性检查点不足或不可达')
            self.candidate=self._table(); self.table_ready=True; self._queue((11,19),'verify'); return self._move_next()
        verification=[s for s in self.samples if s['stage']=='verify' and not s.get('superseded')]
        expected=1 if self.simple else 2
        if len(verification)<expected: return self._fail('验证点不足或不可达')
        if any(np.linalg.norm(s['residual_xy_mm']) > self.cfg['verify_tol_mm'] for s in verification):
            if self.candidate['type']=='constant' and not getattr(self,'in_b',False):
                for s in verification: s['superseded']=True
                self.simple=False; self._begin_b(); return self._move_next()
            return self._fail('补偿后仍超差')
        self.phase='confirm'; return 'confirm'

    def _begin_b(self):
        self.in_b=True; self.candidate=None
        existing={s['target'] for s in self._fit_samples()}
        self._queue([i for i in B if f'T{i:02d}' not in existing],'B')
        self._queue((2,29,12),'repeat')

    def result(self):
        return dict(self.candidate, samples=self.samples, infield=self.edges, warnings=self.warnings,
                    skipped=self.skipped, decision=self.decision,
                    verification=[s for s in self.samples if s['stage']=='verify' and not s.get('superseded')])

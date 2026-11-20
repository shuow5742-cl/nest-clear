"""通信联调的假目标及真实视觉替换适配器。"""
import math
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from autoweaver.worker import AsyncRequestWorker
from nest_clear.guard import GuardError
from nest_clear.plan.scan_path import guard_position
from nest_clear.vision.detector import ModuleTarget, PixelTarget
from nest_clear.vision.fine import FineResult
from nest_clear.vision.hair import HairCandidate
from nest_clear.vision.hair_integration import Observation


def coordinates(config):
    def fine(target, pose):
        point = (target[0], target[1], config.scan.z)
        guard_position(point, config.travel, '联调精定位')
        return point
    def picks(target):
        result = tuple((target[0] + x, target[1] + y, config.pick_depth_z + z) for x, y, z in config.tools)
        for point in result:
            guard_position(point, config.travel, '联调工具坐标')
        return result
    return fine, picks


class World:
    def __init__(self, config, path):
        self.config, self.path = config, path
        self.scenario = config.commtest['scenario']
        rng = random.Random(config.commtest['seed'])
        self.hairs = []
        for index, (x, y, z) in enumerate(path):
            points = []
            # 视野中心 ±12mm 内撒点（让精定位三轴有明显移动），夹在安全边距 5mm 以内
            (xlo, xhi), (ylo, yhi) = config.travel[0], config.travel[1]
            for dx, dy in ((-12,-12), (-12,12), (12,-12), (12,12)):
                point = (min(max(x + dx + rng.uniform(-2,2), xlo + 5), xhi - 5),
                         min(max(y + dy + rng.uniform(-2,2), ylo + 5), yhi - 5), z)
                if any(not lo + 5 <= v <= hi - 5 for v, (lo, hi) in zip(point[:2], config.travel[:2])):
                    raise GuardError('COMMTEST-POINT', f'假点 {point} 不满足安全边距', '联调启动', '检查扫描点与行程。')
                if any(math.dist(point[:2], other[:2]) < math.dist(point[:2], (x,y)) for other in path if other != (x,y,z)):
                    raise GuardError('COMMTEST-POINT', f'假点 {point} 超出视野责任区', '联调启动', '检查扫描路径。')
                for xyz in coordinates(config)[1](point): guard_position(xyz, config.travel, '联调假点')
                points.append(point)
            self.hairs.append(points)
        self.done = set()
        self.failed = set()
        self.processed = set()
        self.rng = rng

    def result(self, payload):
        cycle = payload['cycle']; view = cycle.view_index
        for attempt in cycle.attempts_ledger:
            key = attempt['attempt_id']
            if attempt['done'] and key not in self.processed:
                self.processed.add(key)
                identity = attempt['identity']
                point = min(self.hairs[attempt['view_index']], key=lambda p: math.dist(p[:2], identity[:2]))
                index = self.hairs[attempt['view_index']].index(point)
                # tool_change：首视野 3 根挑不掉→触发换工具判定回 10；mixed：视野 2、4 各 1 根挑不掉（红框），不足以触发判定，仍以 9 结束
                persistent = (self.scenario == 'tool_change' and attempt['view_index'] == 0 and index < 3) or (self.scenario == 'mixed' and attempt['view_index'] in (1, 3) and index == 1)
                if not persistent: self.done.add((attempt['view_index'], index))
        observations = []
        for i, point in enumerate(self.hairs[view]):
            if (view, i) in self.done: continue
            px = PixelTarget(0., 0., 1.)
            candidate = HairCandidate((0.,0.), None, (), (-2.,-2.,2.,2.), (0.,0.), 0, 0, None, None, 'stub', 'stub')
            observations.append(Observation(ModuleTarget(px, point), point, candidate, 1., 10. - i, None))
        if payload['work_action'] == 'coarse':
            return {'observations': observations}
        target = cycle.current_target().target.xyz
        idx = self.hairs[view].index(target)
        fail = self.scenario in ('fine_fail','mixed') and idx == 0 and view in (0, 2, 4) and (view, idx) not in self.failed
        if fail:
            self.failed.add((view, idx))
            return {'observations': observations, 'fine_result': FineResult('fine_empty' if len(self.failed) % 2 else 'fine_point_outside')}
        angle = self.rng.random() * math.tau
        xyz = (target[0] + .2 * math.cos(angle), target[1] + .2 * math.sin(angle), target[2])
        return {'observations': observations, 'fine_result': FineResult('ok', target_xyz=xyz, identity=target)}


class VisionAdapter(AsyncRequestWorker):
    name = 'vision'
    def __init__(self, real, world, events):
        super().__init__(); self.real, self.world, self.events = real, world, events
        self.cycle = None
        self._pool = ThreadPoolExecutor(1, thread_name_prefix='commtest-real-vision'); self._pending = None
    def drain(self, timeout=30):
        """等待影子真实视觉算完（测试与收尾用）。"""
        if self._pending is not None: self._pending.result(timeout)
    def on_stop(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
        getattr(super(), 'on_stop', lambda: None)()
    def update_calibration(self, calibration):
        self.real.update_calibration(calibration)
    def _shadow(self, payload, fields):
        """后台线程里完整跑一次真实视觉，结果只落账，不影响应答。"""
        start = time.perf_counter()
        try:
            result = self.real.execute(payload)
            obs = result.get('observations', [])
            fields.update(status='ok' if obs else 'empty', count=len(obs), candidates=[
                {'pixel': tuple(float(v) for v in o.candidate.pick_uv),
                 'module': tuple(float(v) for v in o.target.xyz)} for o in obs],
                fine_status=getattr(result.get('fine_result'), 'status', None))
        except Exception as exc:
            fields.update(status='error', count=0, candidates=[], exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(limit=8))
        fields['elapsed_ms'] = (time.perf_counter() - start) * 1000
        self.events.emit('real_detection', **fields)

    def execute(self, payload):
        # 真实视觉在独立单线程里影子运行：耗时再长也不拖住 PLC 应答；上一张未算完时本张记 skipped_busy
        fields = {'stage': payload['work_action'], 'capture_id': payload.get('capture_id')}
        if self._pending is not None and not self._pending.done():
            self.events.emit('real_detection', **fields, status='skipped_busy', count=0, candidates=[], elapsed_ms=0.0)
        else:
            self._pending = self._pool.submit(self._shadow, dict(payload), fields)
        result = None
        if payload['cycle'] is not self.cycle:
            if self.cycle is not None:
                self.world = World(self.world.config, self.world.path)
            self.cycle = payload['cycle']
        fake = self.world.result(payload)
        fake['elapsed_ms'] = 0.0
        fake['um_per_px'] = result.get('um_per_px') if result else None
        fake['real_observations'] = result.get('observations', []) if result else []
        archive = getattr(self.events, 'archive', None)
        if isinstance(archive, ArchiveAdapter):
            point = (fake['observations'][0].target.xyz if payload['work_action'] == 'coarse' and fake['observations'] else getattr(fake.get('fine_result'), 'target_xyz', None))
            archive.remember(payload.get('capture_id'), payload['image'], fake['real_observations'], point, payload['pose'], self.real.calibration, payload['cycle'], payload['work_action'])
        return fake


def worker(real, config, path, events):
    world = World(config, path)
    if getattr(events, 'archive', None) is not None:
        events.archive = ArchiveAdapter(events.archive)
    return VisionAdapter(real, world, events)


def resolve_camera():
    from nest_clear.tools.list_cameras import list_cameras
    try:
        devices = list_cameras()
        serials = [str(d.get('sn', d.get('serial_number', ''))) for d in devices]
        if len(serials) != 1 or not serials[0]:
            raise ValueError(f'需要恰好一台带 SN 的相机，发现 {len(serials)} 台：{serials}')
        return serials[0]
    except Exception as exc:
        raise GuardError('COMMTEST-CAMERA', str(exc), '自动选择相机', '检查大恒 SDK 和相机连接。') from exc


def unsupported_branch(ctx):
    from autoweaver.motion_policy.nodes.control.sequence import Sequence
    from nest_clear.flow.leaves import Instant, Predicate, PrepareError
    from nest_clear.guard import diagnostic
    class Unsupported(Instant):
        def __init__(self): super().__init__('联调不支持的功能码')
        def perform(self):
            text = '通信联调模式不支持换针对刀'
            error = GuardError('COMMTEST-UNSUPPORTED', text, text, text)
            self._blackboard.write('error', str(error), self.name)
            self._blackboard.write('error_diagnostic', diagnostic(error), self.name)
    return Sequence([Predicate('联调不支持', lambda bb: 100 <= bb.read('request_func') <= 131), Unsupported(), PrepareError(ctx.events)])


class ArchiveAdapter:
    """用真实检测框和假点重新绘制联调画面。"""
    def __init__(self, archive):
        self.archive = archive
        self.frames = {}
    def __getattr__(self, name): return getattr(self.archive, name)
    def remember(self, capture_id, image, observations, point, pose, calibration, cycle, stage):
        self.frames.clear()
        self.frames[capture_id] = (image.copy(), observations, point, pose, calibration, cycle, stage)
    def rendered(self, image, scan, task_id, raw_path, preview=False, capture_id=None):
        frame = self.frames.pop(capture_id, None)
        if frame is not None:
            import cv2
            import numpy as np
            from nest_clear.vision.hair_integration import render
            raw, observations, point, pose, cal, cycle, stage = frame
            if stage == "coarse":
                point = cycle.current_target().target.xyz if cycle.state.value == "WAIT_FINE" else None
            elif cycle.state.value != "WAIT_PICK":
                point = None
            image = render(raw, observations, info={'capture_id': capture_id, 'view': scan, 'task_id': task_id})
            if point is not None:
                label = 'FAKE XYZ ' + ','.join(f'{v:.3f}' for v in point)
                try:
                    camera = (np.asarray(point) - (np.asarray(pose) + cal.translation)) @ cal.rotation
                    if camera[2] <= 0: raise ValueError('point behind camera')
                    uv, _ = cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3), cal.K, cal.distortion)
                    u, v = uv.reshape(2)
                    if not np.isfinite([u,v]).all() or not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]): raise ValueError('outside image')
                    xy = (round(u),round(v))
                    cv2.drawMarker(image, xy, (255,0,255), cv2.MARKER_CROSS, 30, 2)
                    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,255), 2)
                except Exception:
                    baseline = min(image.shape[0] - 1, 120)
                    cv2.rectangle(image, (0, max(0, baseline - 22)),
                                  (image.shape[1] - 1, baseline), (32, 32, 32), -1)
                    cv2.putText(image, label, (10, baseline), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,0,255), 2)
        return self.archive.rendered(image, scan, task_id, raw_path, preview, capture_id=capture_id)


def detector_factory(config, calibration, path):
    from nest_clear.workers.vision import HairDetector
    try:
        detector = HairDetector(config.vision.hair_detector, calibration, path[0][2], config.vision.scale_z_bin_mm)
        for z in {p[2] for p in path}: detector.for_z(z)
        return detector
    except Exception as exc:
        error = exc
        class FailedDetector:
            def detect(self, image): raise error
        return FailedDetector()


def branches():
    from nest_clear.flow.dispatch import BRANCHES
    from nest_clear.flow.modes import Mode
    result = dict(BRANCHES)
    for code in range(100, 132):
        result[code] = (frozenset(Mode), unsupported_branch)
    return result


class Assembly:
    def prepare(self, config, path):
        World(config, path)
    def detector(self, config, calibration, path):
        return detector_factory(config, calibration, path)
    def camera_sn(self, config):
        return resolve_camera() if config.camera_sn == 'auto' else config.camera_sn
    def vision(self, real, config, path, events):
        return worker(real, config, path, events)
    def branches(self):
        return branches()
    def scale_error(self, exc, z):
        return {'z_mm': z, 'error': str(exc)}


def assembly_factory():
    return Assembly()

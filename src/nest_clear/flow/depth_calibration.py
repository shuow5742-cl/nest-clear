"""160/161 下针深度标定第一阶段的 PLC 请求入口。"""

import logging
import math

from autoweaver.motion_policy.nodes.node import Status

from nest_clear.depthcal.record import DepthRecord
from nest_clear.depthcal.session import Session
from nest_clear.guard import GuardError, diagnostic
from nest_clear.preview.control import REQUEST_CAMERAS, close_previews_for
from .calibration import change_mode
from .modes import Mode
from .sampling import DeviceSampling
from .tool_offset import reader

RESERVED_CODES = frozenset((170, 171, 180, 181, 190, 191))
DEPTH_CODES = RESERVED_CODES | {160}


class DepthCalibration(DeviceSampling):
    def __init__(self, ctx):
        super().__init__('下针深度标定')
        self.ctx = ctx
        self.settle_config = ctx.config.depth_calibration if ctx.config is not None else {}

    def reply(self, code, coords=None):
        bb = self._blackboard
        bb.write('reply_func', code, self.name)
        bb.write('reply_coordinates', coords or {}, self.name)
        bb.write('reply_tool', None, self.name)
        return Status.SUCCESS

    def finish(self, outcome, details=None):
        bb = self._blackboard
        session = bb.read('depth_session', None)
        if session:
            session.record.finish(session, outcome, details)
        if bb.read('mode') == Mode.DEPTH_CALIB:
            change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write('depth_session', None, self.name)
        bb.write('depth_image', None, self.name)
        self.frames = []
        self.ctx.events.abort_depthcal = None
        self.ctx.events.depth_reply_committed = None

    def interrupt(self, reason):
        details = diagnostic(reason, code='DEPTH-INTERRUPTED', where='主程序停止',
                             hint='恢复 PLC 正常状态后从 160 重新发起。')
        self.publish('failed', **details)
        self.finish('aborted', details)

    def committed(self, func, details):
        bb = self._blackboard
        session = bb.read('depth_session')
        coordinates = bb.read('reply_coordinates') if func == 160 else {}
        reply = {'task_id': bb.read('task_id'), 'func': func, 'coordinates': coordinates}
        session.replies.append(reply)
        session.record.interaction('reply', {**reply, 'diagnostic': details or None,
                                            'time': self.ctx.events.clock().isoformat()})
        if func == 161:
            self.publish('done', message='完成')
            self.finish('done')
        elif func == 999:
            self.publish('failed', **details)
            self.finish('failed', details)

    def record_request(self, session):
        bb = self._blackboard
        row = {'task_id': bb.read('task_id'), 'func': bb.read('request_func'),
               'time': self.ctx.events.clock().isoformat()}
        session.requests.append(row)
        session.record.interaction('request', row)

    def publish(self, phase=None, **details):
        session = self._blackboard.read('depth_session', None)
        data = {}
        if session:
            data = {'phase': session.phase, 'message': session.message,
                    'label': session.label, 'z': session.actual[2] if session.actual else None,
                    'point': min(session.point_index + 1, len(session.plan)), 'total': len(session.plan),
                    'plan': session.plan, 'plane': session.plane, 'initial_plane': session.initial_plane,
                    'results': session.results, 'feasibility': session.feasibility,
                    'metric': session.cfg['metric'], 'method': session.cfg['peak_method'],
                    'dof_ratio': session.cfg['dof_ratio'],
                    'fine_plan': session.fine_plan,
                    'coarse_fit': session.coarse_fit,
                    'coarse_dof': getattr(session, 'coarse_dof', None),
                    'current_fits': list(session.fits),
                    'dof_summary': getattr(session, 'dof_summary', None),
                    'session': session.record.directory.name, 'sample': session.last_sample,
                    'abort_requested': session.abort_requested}
            session.last_sample = None
        if phase:
            data['phase'] = phase
        self.ctx.events.emit('depthcal_state', **(data | details))

    def fail(self, exc):
        bb = self._blackboard
        details = diagnostic(exc, code='DEPTH-FAILED', where='下针深度标定',
                             hint='检查菲林、相机和 PLC 实时坐标。')
        bb.write('reply_diagnostic', details, self.name)
        if bb.read('depth_session', None) is None:
            self.publish('failed', **details)
            self.finish('failed', details)
        return self.reply(999)

    def on_start(self):
        bb = self._blackboard
        bb.write('depth_request', True, self.name)
        func, mode = bb.read('request_func'), bb.read('mode')
        session = bb.read('depth_session', None)
        if session:
            self.record_request(session)
        try:
            if func in RESERVED_CODES:
                raise GuardError('DEPTH-UNIMPLEMENTED', '下针深度标定 170/180/190 尚未实现',
                                 'PLC 请求', '本阶段使用 160 启动料盘对焦测试。')
            if func != 160 or mode not in (Mode.IDLE, Mode.DEPTH_CALIB):
                raise GuardError('DEPTH-MODE', f'当前模式 {mode.value} 不允许功能码 {func}',
                                 'PLC 请求', '空闲时从 160 发起料盘对焦测试。')
            if self.ctx.config.depth_calibration['camera_z_min_mm'] is None:
                raise GuardError('DEPTH-Z-MIN', '未配置相机 Z 下限 depth_calibration.camera_z_min_mm（用户现场给出）',
                                 '160 会话启动', '现场确认两把工具与料盘间隙后填写 Z 下限。')
            return self.start_request()
        except Exception as exc:
            return self.fail(exc)

    def start_request(self):
        bb, ctx = self._blackboard, self.ctx
        session = bb.read('depth_session', None)
        if session is None:
            if ctx.calibration is None or (ctx.calibration.placeholder and
                                            not ctx.config.vision.allow_placeholder_calibration):
                raise GuardError('DEPTH-CALIBRATION', '缺少有效的主相机标定',
                                 '160 会话启动', '请先完成主相机标定。')
            session = Session(ctx.config, ctx.calibration, DepthRecord(ctx.events))
            session.record.begin()
            bb.write('depth_session', session, self.name)
            ctx.events.abort_depthcal = self.interrupt
            ctx.events.depth_reply_committed = self.committed
            self.record_request(session)
            change_mode(bb, ctx, Mode.DEPTH_CALIB, self.name)
        else:
            if session.abort_requested:
                return self.abort()
            close_previews_for(ctx.events, REQUEST_CAMERAS.get(bb.read('request_func'), ()), 'plc_request',
                               bb.read('task_id'), bb.read('request_func'), mode=bb.read('mode').value)
        self.call = None
        self.samples, self.round, self.next_read = [], 0, 0.
        self.frames = []
        self.capturing = False
        self.publish()
        return Status.RUNNING

    def abort(self):
        return self.fail(GuardError('DEPTH-ABORTED', '用户放弃料盘对焦测试', '下针深度标定',
                                    '可从 160 重新发起。'))

    def capture(self):
        self.capturing = True
        self.start_call('camera', {'stage': 'depthcal', 'task_id': self._blackboard.read('task_id')},
                        'depth_image')
        return Status.RUNNING

    def on_running(self):
        bb = self._blackboard
        session = bb.read('depth_session', None)
        if session is None:
            return self.fail('下针深度标定会话已中断')
        try:
            if reader(self.ctx).poll({'depthcal_abort'}, defer_unallowed=True):
                return self.abort()
            if not self.capturing:
                actual = self.settle()
                if actual is None:
                    return Status.RUNNING
                if actual is False:
                    raise GuardError('DEPTH-SETTLE', 'PLC 实时坐标未稳定', session.where,
                                     '检查到位时序、模组振动与 jitter_tol_mm。')
                session.arrived(actual)
                if session.phase == 'overshoot':
                    return self.move_reply(session.after_overshoot())
                return self.capture()
            status = self.tick_call()
            if status == Status.RUNNING:
                return status
            if status == Status.FAILURE:
                return self.fail(bb.read('error'))
            image, metadata = bb.read('depth_image')
            exposure = metadata.get('exposure_us')
            if exposure is not None:
                previous = getattr(session, 'exposure_us', exposure)
                if not math.isclose(exposure, previous, rel_tol=1e-6):
                    raise GuardError('DEPTH-EXPOSURE', f'扫描曝光从 {previous} 变为 {exposure} µs',
                                     session.where, '检查相机固定曝光及光照后重试。')
                session.exposure_us = exposure
            self.frames.append(image)
            if len(self.frames) < session.frame_count:
                return self.capture()
            if session.phase == 'peak' and session.point_index + 1 == len(session.plan):
                self.publish('plane', message='平面判定')
            coords = session.advance(self.frames)
            self.frames = []
            if coords is not None:
                return self.move_reply(coords)
            session.record.results(session)
            return self.reply(161)
        except GuardError as exc:
            return self.fail(exc)
        except Exception as exc:
            logging.getLogger(__name__).exception('下针深度标定异常')
            return self.fail(exc)

    def move_reply(self, coords):
        self.publish()
        return self.reply(160, {'scan_xyz': coords})

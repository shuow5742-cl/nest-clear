"""160/161 下针深度标定第一阶段的 PLC 请求入口。"""

import logging
import math
import time

from autoweaver.motion_policy.nodes.node import Status

from nest_clear.depthcal.exposure import (
    load_depthcal_exposure, save_depthcal_exposure, validate_depthcal_exposure,
)
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


def handle_depthcal_exposure_command(events, config, params, session=None):
    """处理界面发来的下针深度标定曝光模式/手动值设置命令并持久化到 runs/state。"""
    cfg = config.depth_calibration if config is not None else None
    seq = params.get('_seq') if isinstance(params, dict) else None
    try:
        mode, manual_val = validate_depthcal_exposure(
            params.get('mode') if isinstance(params, dict) else None,
            params.get('manual_exposure_us') if isinstance(params, dict) else None,
            cfg,
        )
    except ValueError as exc:
        events.emit(
            'depthcal_exposure_state',
            exposure_error=str(exc),
            exposure_command_message=str(exc),
            exposure_command_seq=seq,
        )
        return False
    save_depthcal_exposure(events.directory, mode, manual_val, seq=seq, clock=events.clock)
    pending_next = False
    if session is not None:
        session.set_desired_exposure(mode, manual_val)
        pending_next = bool(session.exposure_locked)
    mode_label = '自动' if mode == 'auto' else f'手动 {manual_val:.0f} µs'
    msg = f'已保存曝光设置（{mode_label}，下一测点生效）' if pending_next else f'已保存曝光设置（{mode_label}）'
    events.emit(
        'depthcal_exposure_state',
        exposure_mode=mode,
        manual_exposure_us=manual_val,
        exposure_error='',
        exposure_command_message=msg,
        exposure_pending_next_point=pending_next,
        exposure_command_seq=seq,
    )
    return True


class DepthCalibration(DeviceSampling):
    def __init__(self, ctx):
        super().__init__('下针深度标定')
        self.ctx = ctx
        self.settle_config = ctx.config.depth_calibration if ctx.config is not None else {}
        self.init_exp_stage = None
        self.restore_stage = None
        self.pending_exit = None

    def reply(self, code, coords=None):
        bb = self._blackboard
        bb.write('reply_func', code, self.name)
        bb.write('reply_coordinates', coords or {}, self.name)
        bb.write('reply_tool', None, self.name)
        return Status.SUCCESS

    def _restore_sync(self, session):
        """在中断或收尾兜底路径同步恢复主相机曝光；失败时记录 error 并返回错误文本。"""
        if (session is None or getattr(session, 'exposure_restored', False)
                or getattr(session, 'exposure_restore_error', None)):
            return None
        target_exp = getattr(session, 'initial_exposure_us', None)
        if target_exp is None and self.ctx.config is not None:
            target_exp = self.ctx.config.exposure_us
        target_auto = getattr(session, 'initial_exposure_auto', None)
        camera = getattr(self.ctx.events, 'main_camera', None)
        if camera is None or (target_exp is None and target_auto is None):
            return None
        try:
            camera.execute({
                'stage': 'depthcal',
                'operation': 'restore_exposure_state',
                'exposure_us': target_exp,
                'exposure_auto': target_auto,
            })
            session.exposure_restored = True
            return None
        except Exception as exc:
            err = str(exc) or type(exc).__name__
            session.exposure_restore_error = err
            self.ctx.events.emit(
                'error',
                operation='depthcal_restore_exposure',
                reason=f'主相机曝光恢复失败，请重启程序：{err}',
            )
            return err

    def finish(self, outcome, details=None):
        bb = self._blackboard
        session = bb.read('depth_session', None)
        if session:
            self._restore_sync(session)
            session.record.finish(session, outcome, details)
        if bb.read('mode') == Mode.DEPTH_CALIB:
            change_mode(bb, self.ctx, Mode.IDLE, self.name)
        bb.write('depth_session', None, self.name)
        bb.write('depth_image', None, self.name)
        self.frames = []
        self.ae_stage = None
        self.init_exp_stage = None
        self.restore_stage = None
        self.pending_exit = None
        self.last_capture_meta = None
        self.ctx.events.abort_depthcal = None
        self.ctx.events.depth_reply_committed = None

    def interrupt(self, reason):
        bb = self._blackboard
        session = bb.read('depth_session', None)
        details = diagnostic(reason, code='DEPTH-INTERRUPTED', where='主程序停止',
                             hint='恢复 PLC 正常状态后从 160 重新发起。')
        restore_err = self._restore_sync(session)
        if restore_err:
            details = {
                **details,
                'what': f"{details.get('what', '')}；主相机曝光恢复失败，请重启程序（{restore_err}）".lstrip('；'),
                'hint': f"{details.get('hint', '')} 主相机曝光恢复失败，请重启程序。".strip(),
            }
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
                    'results': session.results,
                    'failed_points': getattr(session, 'failed_points', []),
                    'skip_summary': getattr(session, 'skip_summary', None),
                    'feasibility': session.feasibility,
                    'metric': session.cfg['metric'], 'method': session.cfg['peak_method'],
                    'dof_ratio': session.cfg['dof_ratio'],
                    'fine_plan': session.fine_plan,
                    'coarse_fit': session.coarse_fit,
                    'coarse_dof': getattr(session, 'coarse_dof', None),
                    'coarse_bounds': getattr(session, 'coarse_bounds', None),
                    'current_fits': list(session.fits),
                    'dof_summary': getattr(session, 'dof_summary', None),
                    'arrival_summary': getattr(session, 'arrival_summary', None),
                    'exposure_mode': getattr(session, 'desired_exposure_mode', 'auto'),
                    'manual_exposure_us': getattr(session, 'desired_manual_exposure_us', None),
                    'current_exposure_us': getattr(session, 'point_exposure_us', None),
                    'point_exposure_mode': getattr(session, 'point_exposure_mode', None),
                    'roi_mean': getattr(session, 'live_roi_mean', None),
                    'roi_p99': getattr(session, 'live_roi_p99', None),
                    'exposure_warning': getattr(session, 'point_exposure_warning', None),
                    'exposure_min_us': session.cfg.get('exposure_min_us'),
                    'exposure_max_us': session.cfg.get('exposure_max_us'),
                    'session': session.record.directory.name, 'sample': session.last_sample,
                    'abort_requested': session.abort_requested}
            session.last_sample = None
        if phase:
            data['phase'] = phase
        self.ctx.events.emit('depthcal_state', **(data | details))

    def _begin_restore(self, reply_func, details=None):
        bb = self._blackboard
        session = bb.read('depth_session', None)
        if session is None or getattr(session, 'exposure_restored', False):
            if details is not None:
                bb.write('reply_diagnostic', details, self.name)
            return self.reply(reply_func)
        target_exp = getattr(session, 'initial_exposure_us', None)
        if target_exp is None and self.ctx.config is not None:
            target_exp = self.ctx.config.exposure_us
        target_auto = getattr(session, 'initial_exposure_auto', None)
        if target_exp is None and target_auto is None:
            if details is not None:
                bb.write('reply_diagnostic', details, self.name)
            return self.reply(reply_func)
        self.ae_stage = None
        self.init_exp_stage = None
        self.pending_exit = (reply_func, details)
        self.restore_stage = 'restoring'
        self.start_call(
            'camera',
            {
                'stage': 'depthcal',
                'operation': 'restore_exposure_state',
                'exposure_us': target_exp,
                'exposure_auto': target_auto,
                'task_id': bb.read('task_id'),
            },
            'depth_restore_exposure',
        )
        return Status.RUNNING

    def fail(self, exc):
        bb = self._blackboard
        details = diagnostic(exc, code='DEPTH-FAILED', where='下针深度标定',
                             hint='检查菲林、相机和 PLC 实时坐标。')
        if bb.read('depth_session', None) is None:
            bb.write('reply_diagnostic', details, self.name)
            self.publish('failed', **details)
            self.finish('failed', details)
            return self.reply(999)
        return self._begin_restore(999, details)

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
        is_new_session = session is None
        if is_new_session:
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
        self.arrival_started = None
        self.frames = []
        self.capturing = False
        self.ae_stage = None
        self.init_exp_stage = None
        self.restore_stage = None
        self.pending_exit = None
        self.last_capture_meta = None
        if is_new_session:
            self.init_exp_stage = 'getting'
            self.start_call(
                'camera',
                {'stage': 'depthcal', 'operation': 'get_exposure_state', 'task_id': bb.read('task_id')},
                'depth_init_exposure',
            )
        self.publish()
        return Status.RUNNING

    def settle(self, expected=None):
        """非阻塞滑动窗口等待 PLC 实时坐标稳定且到位；超时后取最后窗口均值继续。"""
        cfg = self.settle_config
        now = time.monotonic()
        if self.arrival_started is None:
            self.arrival_started = now
        if self.call:
            status = self.tick_call()
            if status == Status.FAILURE:
                raise ValueError('读取实时坐标失败：' + str(self._blackboard.read('error')))
            if status == Status.RUNNING:
                return None
            value = self._blackboard.read('stable_actual')
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError('实时坐标无效')
            self.samples.append(tuple(float(x) for x in value))
            now = time.monotonic()
            self.next_read = now + cfg['settle_window_s'] / (cfg['settle_samples'] - 1)
        n = cfg['settle_samples']
        jitter_tol = cfg['jitter_tol_mm']
        pos_tol = cfg['position_tolerance_mm']
        timeout_s = cfg['arrival_timeout_s']
        timed_out = self.call is None and bool(self.samples) and (now - self.arrival_started >= timeout_s)
        if len(self.samples) >= n or timed_out:
            window = self.samples[-n:]
            ptp = tuple(max(s[i] for s in window) - min(s[i] for s in window) for i in range(3))
            mean = tuple(sum(s[i] for s in window) / len(window) for i in range(3))
            stable = len(window) >= n and all(r <= jitter_tol for r in ptp)
            in_tol = (expected is None) or all(abs(m - e) <= pos_tol for m, e in zip(mean, expected))
            if (stable and in_tol) or timed_out:
                return mean, {
                    'stable': stable,
                    'ptp': ptp,
                    'elapsed_s': max(0.0, now - self.arrival_started),
                    'samples': len(self.samples),
                    'timed_out': not (stable and in_tol),
                }
        if self.call is None and now >= self.next_read:
            self.start_call('plc', {'operation': 'actual'}, 'stable_actual')
        return None

    def abort(self):
        return self.fail(GuardError('DEPTH-ABORTED', '用户放弃料盘对焦测试', '下针深度标定',
                                    '可从 160 重新发起。'))

    def capture(self):
        self.capturing = True
        self.start_call('camera', {'stage': 'depthcal', 'task_id': self._blackboard.read('task_id')},
                        'depth_image')
        return Status.RUNNING

    def _start_set_exposure(self, target_us):
        self.ae_stage = 'setting'
        self.capturing = True
        self.start_call(
            'camera',
            {'stage': 'depthcal', 'operation': 'set_exposure', 'exposure_us': float(target_us),
             'task_id': self._blackboard.read('task_id')},
            'depth_exposure_res',
        )
        return Status.RUNNING

    def _start_ae_capture(self):
        self.ae_stage = 'capturing'
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
            if self.restore_stage == 'restoring':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return status
                reply_func, details = self.pending_exit
                self.restore_stage = None
                self.pending_exit = None
                if status == Status.FAILURE:
                    err = str(bb.read('error') or '未知错误')
                    session.exposure_restore_error = err
                    self.ctx.events.emit(
                        'error',
                        task_id=bb.read('task_id'),
                        operation='depthcal_restore_exposure',
                        reason=f'主相机曝光恢复失败，请重启程序：{err}',
                    )
                    if details is None:
                        details = diagnostic(
                            GuardError(
                                'DEPTH-EXPOSURE-RESTORE',
                                f'主相机曝光恢复失败，请重启程序（{err}）',
                                '会话结束恢复曝光',
                                '请重启主程序与父程序，避免后续流程采图曝光异常。',
                            )
                        )
                        reply_func = 999
                    else:
                        details = {
                            **details,
                            'what': f"{details.get('what', '')}；主相机曝光恢复失败，请重启程序（{err}）".lstrip('；'),
                            'hint': f"{details.get('hint', '')} 主相机曝光恢复失败，请重启程序。".strip(),
                        }
                    bb.write('reply_diagnostic', details, self.name)
                    return self.reply(reply_func)
                session.exposure_restored = True
                if details is not None:
                    bb.write('reply_diagnostic', details, self.name)
                return self.reply(reply_func)
            cmd = reader(self.ctx).poll({'depthcal_abort', 'depthcal_exposure'}, defer_unallowed=True)
            if cmd and cmd[0] == 'depthcal_abort':
                return self.abort()
            if cmd and cmd[0] == 'depthcal_exposure':
                handle_depthcal_exposure_command(self.ctx.events, self.ctx.config, cmd[1], session=session)
                self.publish()
            if self.init_exp_stage == 'getting':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return status
                self.init_exp_stage = None
                fallback_exp = self.ctx.config.exposure_us if self.ctx.config is not None else None
                if status == Status.FAILURE:
                    session.initial_exposure_us = float(fallback_exp) if fallback_exp is not None else None
                    session.initial_exposure_auto = None
                else:
                    _, meta = bb.read('depth_init_exposure')
                    exp = meta.get('exposure_us') if isinstance(meta, dict) else None
                    auto = meta.get('exposure_auto') if isinstance(meta, dict) else None
                    if exp is None or not math.isfinite(exp) or exp <= 0:
                        exp = fallback_exp
                    session.initial_exposure_us = float(exp) if exp is not None else None
                    session.initial_exposure_auto = auto
                self.publish()
                if (session.initial_exposure_auto not in (None, 0)
                        and session.initial_exposure_us is not None):
                    self.init_exp_stage = 'fixing'
                    self.start_call(
                        'camera',
                        {'stage': 'depthcal', 'operation': 'set_exposure',
                         'exposure_us': float(session.initial_exposure_us), 'task_id': bb.read('task_id')},
                        'depth_init_exposure_fix',
                    )
                    return Status.RUNNING
            if self.init_exp_stage == 'fixing':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return status
                if status == Status.FAILURE:
                    return self.fail(bb.read('error'))
                self.init_exp_stage = None
            if self.ae_stage is not None:
                status = self.tick_call()
                if status == Status.RUNNING:
                    return status
                if status == Status.FAILURE:
                    return self.fail(bb.read('error'))
                if self.ae_stage == 'setting':
                    return self._start_ae_capture()
                image, metadata = bb.read('depth_image')
                exp_val = metadata.get('exposure_us') if isinstance(metadata, dict) else None
                op, val = session.step_exposure(image, exp_val)
                if op == 'set_exposure':
                    self.publish()
                    return self._start_set_exposure(val)
                self.ae_stage = None
                self.capturing = False
                self.frames = []
                if val is not None:
                    return self.move_reply(val)
                session.record.results(session)
                return self._begin_restore(161)
            if not self.capturing:
                settled = self.settle(session.expected)
                if settled is None:
                    return Status.RUNNING
                actual, arrival_info = settled
                entry = session.arrived(actual, arrival_info, task_id=bb.read('task_id'))
                if session.phase in ('coarse_overshoot', 'overshoot'):
                    return self.move_reply(session.after_overshoot())
                if entry['status'] != 'ok':
                    self.publish()
                return self.capture()
            status = self.tick_call()
            if status == Status.RUNNING:
                return status
            if status == Status.FAILURE:
                return self.fail(bb.read('error'))
            image, metadata = bb.read('depth_image')
            self.last_capture_meta = metadata if isinstance(metadata, dict) else {}
            exposure = self.last_capture_meta.get('exposure_us')
            if session.exposure_locked and exposure is not None and session.point_exposure_us is not None:
                if not math.isclose(exposure, session.point_exposure_us, rel_tol=1e-6):
                    raise GuardError('DEPTH-EXPOSURE',
                                     f'扫描曝光从 {session.point_exposure_us} 变为 {exposure} µs',
                                     session.where, '检查相机固定曝光及光照后重试。')
            self.frames.append(image)
            if len(self.frames) < session.frame_count:
                return self.capture()
            if session.phase == 'peak' and session.point_index + 1 == len(session.plan):
                self.publish('plane', message='平面判定')
            coords = session.advance(self.frames, metadata=self.last_capture_meta)
            self.frames = []
            if session.phase == 'exposure':
                self.publish()
                return self._start_set_exposure(coords)
            if coords is not None:
                return self.move_reply(coords)
            session.record.results(session)
            return self._begin_restore(161)
        except GuardError as exc:
            return self.fail(exc)
        except Exception as exc:
            logging.getLogger(__name__).exception('下针深度标定异常')
            return self.fail(exc)

    def move_reply(self, coords):
        self.publish()
        return self.reply(160, {'scan_xyz': coords})

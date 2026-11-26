"""新协议工具偏移标定：主相机三角定位标定尖（90→91）+ 对刀相机对准（100/110→111/121）+ 核对保存（130）。"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml
from autoweaver.motion_policy.nodes.node import Status, TreeNode

from nest_clear.guard import GuardError, diagnostic
from nest_clear.tooloffset.commands import CommandReader
from nest_clear.tooloffset.store import save_confirmed
from .calibration import change_mode
from .leaves import CallWorker
from .modes import Mode


def reader(ctx):
    """获取或初始化界面命令读取器。"""
    if not hasattr(ctx.events, 'command_reader'):
        ctx.events.command_reader = CommandReader(ctx.events.directory, ctx.events)
    return ctx.events.command_reader


def finish(ctx, bb, outcome, reason=None):
    """结束工具偏移会话、写 session_record.json 并恢复空闲模式。"""
    session = bb.read('tooloffset_session', None)
    if session:
        record = {
            'tool': session.get('tool'),
            'plc_tool_id': session.get('plc_tool_id', 1),
            'start_time': session['start_time'],
            'end_time': ctx.events.clock().isoformat(),
            'outcome': outcome,
            'reason': reason,
            'camera_calibration_sha256': session.get('hash'),
            'requests': session.get('requests', []),
            'replies': session.get('replies', []),
            'commands': session.get('commands', []),
            'triangulation': session.get('triangulation_record'),
            'alignment': session.get('alignment_record'),
            'p0_xyz_mm': session.get('p0'),
            'p_final_xyz_mm': session.get('p_final'),
            'plc_final_xyz_mm': session.get('plc_final'),
            'offset_xy_mm': session.get('offset_xy'),
            'shots': session.get('shots', 0),
        }
        if ctx.events.run_directory is not None:
            path = ctx.events.run_directory / 'tooloffset' / session['directory'] / 'session_record.json'
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            except (OSError, TypeError, ValueError) as exc:
                ctx.events.emit('error', operation='tooloffset_record', reason=str(exc))
        ctx.events.emit('tooloffset_' + outcome, **record)
    if bb.read('mode') == Mode.TOOL_OFFSET:
        change_mode(bb, ctx, Mode.IDLE, '结束工具偏移标定')
    bb.write('tooloffset_session', None, '结束工具偏移标定')
    ctx.events.abort_tooloffset = None


def _existing(ctx, path, digest):
    """读取现有两把工具偏移标定结果及标定失效标记。"""
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError, yaml.YAMLError):
        data = {}
    return {
        f'tool{i}': {
            **(data.get(f'tool{i}') or {}),
            'calibration_changed': bool(data.get(f'tool{i}') and data[f'tool{i}'].get('camera_calibration_sha256') != digest),
            'current_tools_xy': list(ctx.config.tools[i - 1][:2]),
        }
        for i in (1, 2)
    }


def _failure_fields(reason, code='TOOLOFFSET-FAILED', where='工具偏移标定', hint='检查标定尖、相机视野与 PLC 坐标。'):
    """拆分结构化失败诊断，供状态与界面直接展示现象/位置/建议。"""
    diag = diagnostic(reason, code=code, where=where, hint=hint)
    return {'failure_code': diag['reason_code'], 'failure_what': diag['what'],
            'failure_where': diag['where'], 'failure_hint': diag['hint']}


class ToolOffset(TreeNode):
    """工具偏移标定行为树节点；同时为 PickAccuracy 提供共享的采图/稳定采样基类方法。"""

    def __init__(self, ctx):
        super().__init__('工具偏移标定')
        self.ctx = ctx
        self.call = None

    def publish(self, phase, message='', **extra):
        """向事件总线与主状态发布工具偏移实时进度。"""
        s = self._blackboard.read('tooloffset_session', None)
        fields = {
            'phase': phase,
            'message': message,
            'tool': s.get('tool') if s else None,
            'plc_tool_id': self._blackboard.read('plc_tool_id'),
            'command_seq': reader(self.ctx).last_seq,
        }
        if s:
            fields['existing'] = _existing(self.ctx, s['path'], s['hash'])
            fields['abort_requested'] = bool(s.get('abort_requested'))
            if s.get('triangulation_state') is not None:
                fields['triangulation'] = s['triangulation_state']
            if s.get('alignment_state') is not None:
                fields['alignment'] = s['alignment_state']
            if s.get('p0') is not None:
                fields['p0'] = s['p0']
            if s.get('p_final') is not None:
                fields['p_final'] = s['p_final']
            if s.get('offset_xy') is not None:
                fields['offset_xy'] = s['offset_xy']
            if s.get('old_offset_xy') is not None:
                fields['old_offset_xy'] = s['old_offset_xy']
        fields.update(extra)
        self.ctx.events.emit('tooloffset_state', **fields)

    def set_reply(self, code, coords=None, reason=None, *, code_id='TOOLOFFSET-FAILED', where='工具偏移标定', hint='检查标定尖、相机视野与 PLC 坐标。'):
        """设置应答功能码与数据区；40935 由统一应答出口自动写入。"""
        bb = self._blackboard
        s = bb.read('tooloffset_session', None)
        if s is not None:
            s.setdefault('replies', []).append({
                'task_id': bb.read('task_id'),
                'reply_func': code,
                'coordinates': dict(coords or {}),
                'reason': reason,
            })
        bb.write('reply_func', code, self.name)
        bb.write('reply_coordinates', coords or {}, self.name)
        bb.write('reply_tool', None, self.name)
        if reason:
            bb.write('reply_diagnostic', diagnostic(reason, code=code_id, where=where, hint=hint), self.name)
        return Status.SUCCESS

    def fail(self, reason, *, code='TOOLOFFSET-FAILED', where='工具偏移标定', hint='检查标定尖、相机视野与 PLC 坐标。'):
        """新协议业务失败统一结束会话、发布失败诊断并回 99。"""
        diag_fields = _failure_fields(reason, code=code, where=where, hint=hint)
        self.publish('failed', reason, **diag_fields)
        finish(self.ctx, self._blackboard, 'failed', reason)
        return self.set_reply(99, reason=reason, code_id=code, where=where, hint=hint)

    def start_call(self, target, payload, result):
        """发起非阻塞 Worker 调用。"""
        self.call = CallWorker(self.ctx.board, target, payload, result, name='工具偏移设备调用')
        self.call.set_blackboard(self._blackboard)

    def tick_call(self):
        """推进一次 Worker 调用。"""
        status = self.call.tick(self.snapshot, self.tick_ctx)
        if status != Status.RUNNING:
            self.call = None
        return status

    def settle(self):
        """非阻塞等待 PLC 实时坐标在窗口内稳定，取均值。"""
        cfg = self.ctx.config.tool_offset
        if self.call:
            status = self.tick_call()
            if status == Status.FAILURE:
                raise ValueError('读取实时坐标失败：' + str(self._blackboard.read('error')))
            if status == Status.RUNNING:
                return None
            value = self._blackboard.read('tooloffset_actual')
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError('实时坐标无效')
            self.samples.append(value)
            self.next_read = time.monotonic() + cfg['settle_window_s'] / (cfg['settle_samples'] - 1)
        if len(self.samples) >= cfg['settle_samples']:
            a = np.asarray(self.samples)
            if np.all(np.ptp(a, axis=0) <= cfg['jitter_tol_mm']):
                return tuple(float(x) for x in a.mean(axis=0))
            self.round += 1
            if self.round >= 3:
                return False
            self.samples = []
        if self.call is None and time.monotonic() >= self.next_read:
            self.start_call('plc', {'operation': 'actual'}, 'tooloffset_actual')
        return None

    def on_start(self):
        """校验会话与请求顺序，准备稳定采样或 130 核对。"""
        self.call = None
        self.stage = None
        bb, ctx = self._blackboard, self.ctx
        s = bb.read('tooloffset_session', None)
        func = bb.read('request_func')
        task_id = bb.read('task_id')
        try:
            if func == 90 and s is None:
                path = Path(ctx.calibration_path or ctx.config.vision.calibration)
                if ctx.calibration is None or not path.is_file():
                    raise GuardError('CALIBRATION-MISSING', '相机标定文件不存在', '工具偏移标定', '请先完成主相机标定。')
                if getattr(ctx.calibration, 'placeholder', False) and not ctx.config.vision.allow_placeholder_calibration:
                    raise GuardError('CALIBRATION-PLACEHOLDER', '相机标定为占位值，不能执行工具偏移标定', '工具偏移标定', '请先完成主相机标定。')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                now = ctx.events.clock()
                s = {
                    'phase': 'triangulate',
                    'tool': None,
                    'plc_tool_id': bb.read('plc_tool_id'),
                    'path': Path(ctx.config.tool_calibration.directory) / 'tool_offset.yaml',
                    'hash': digest,
                    'shots': 0,
                    'commands': [],
                    'abort_requested': False,
                    'start_time': now.isoformat(),
                    'directory': now.strftime('%Y%m%d_%H%M%S_%f'),
                    'requests': [{'task_id': task_id, 'func': func, 'plc_tool_id': bb.read('plc_tool_id')}],
                    'replies': [],
                    'tri_session': None,
                    'triangulation_state': {'views': [], 'xyz': None, 'target_xy': None, 'max_residual_mm': None, 'max_baseline_ratio': None},
                    'align_session': None,
                    'alignment_state': None,
                    'triangulation_record': None,
                    'alignment_record': None,
                    'p0': None,
                    'p_final': None,
                    'plc_final': None,
                    'offset_xy': None,
                    'old_offset_xy': None,
                }
                bb.write('tooloffset_session', s, self.name)
                change_mode(bb, ctx, Mode.TOOL_OFFSET, self.name)
                ctx.events.abort_tooloffset = lambda reason: finish(ctx, bb, 'aborted', reason)
                self.stage = 'settle_triangulate'
                self.samples, self.round, self.next_read = [], 0, 0.0
                self.publish('triangulate', '三角定位 第 1/≥3 张：读取实时坐标')
                return Status.RUNNING

            if s is None:
                raise GuardError('TOOLOFFSET-NO-SESSION', f'无工具偏移会话时收到 {func}', '工具偏移标定', '请先从功能码 90 发起工具偏移标定。')

            s['requests'].append({'task_id': task_id, 'func': func, 'plc_tool_id': bb.read('plc_tool_id')})
            if s.get('abort_requested'):
                return self.fail('用户放弃标定', code='TOOLOFFSET-ABORTED', where='工具偏移标定', hint='重新从 90 发起标定。')

            if func == 90:
                if s['phase'] != 'triangulate':
                    return self.fail(f"当前处于 {s['phase']} 阶段，却收到功能码 90", code='TOOLOFFSET-ORDER', where='请求顺序检查')
                self.stage = 'settle_triangulate'
                self.samples, self.round, self.next_read = [], 0, 0.0
                shot_no = len(s['triangulation_state']['views']) + 1
                self.publish('triangulate', f'三角定位 第 {shot_no}/≥3 张：检查相机到位')
                return Status.RUNNING

            if func in (100, 110):
                tool = 1 if func == 100 else 2
                if s['phase'] == 'wait_tool_start':
                    s['tool'] = tool
                    s['old_offset_xy'] = list(ctx.config.tools[tool - 1][:2])
                    s['phase'] = 'probing'
                elif s['phase'] in ('probing', 'aligning'):
                    if s['tool'] != tool:
                        return self.fail(f"标定工具 {s['tool']} 过程中收到另一把工具功能码 {func}", code='TOOLOFFSET-TOOL-MISMATCH', where='工具功能码检查')
                else:
                    return self.fail(f"当前处于 {s['phase']} 阶段，却收到功能码 {func}", code='TOOLOFFSET-ORDER', where='请求顺序检查')
                self.stage = 'settle_tool'
                self.samples, self.round, self.next_read = [], 0, 0.0
                label = '读取初始坐标 P0' if s['align_session'] is None else '检查工具到位'
                self.publish(s['phase'], label)
                return Status.RUNNING

            if func == 130:
                if s['phase'] != 'wait_130':
                    return self.fail(f"未完成工具对准（当前阶段 {s['phase']}）却收到 130", code='TOOLOFFSET-ORDER', where='130 完成核对')
                self.start_call('plc', {'operation': 'calibration_final', 'tool': s['tool']}, 'tooloffset_plc_final')
                self.stage = 'verify_130'
                self.publish('wait_130', '读取 PLC 最终坐标并核对')
                return Status.RUNNING

            return self.fail(f'工具偏移会话收到不支持的功能码 {func}', code='TOOLOFFSET-ORDER')
        except GuardError as exc:
            if bb.read('tooloffset_session', None):
                finish(ctx, bb, 'aborted', exc.what)
            bb.write('error', str(exc), self.name)
            bb.write('error_diagnostic', diagnostic(exc, where=self.name), self.name)
            return Status.FAILURE
        except Exception as exc:
            if bb.read('tooloffset_session', None):
                finish(ctx, bb, 'aborted', str(exc))
            bb.write('error', str(exc), self.name)
            bb.write('error_diagnostic', diagnostic(exc, where=self.name), self.name)
            return Status.FAILURE

    def on_running(self):
        """逐拍推进稳定采样、采图检测、三角定位/对准计算及 130 保存。"""
        bb, ctx = self._blackboard, self.ctx
        s = bb.read('tooloffset_session', None)
        if s is None:
            return self.set_reply(99, reason='工具偏移会话已结束')
        try:
            cmd = reader(ctx).poll({'tooloffset_abort', 'abort'})
            if cmd:
                s['commands'].append({'seq': reader(ctx).last_seq, 'command': cmd[0], 'params': cmd[1]})
                s['abort_requested'] = True
                return self.fail('用户放弃标定', code='TOOLOFFSET-ABORTED', where='工具偏移标定', hint='重新从 90 发起标定。')

            if self.stage == 'settle_triangulate':
                actual = self.settle()
                if actual is None:
                    return Status.RUNNING
                if actual is False:
                    return self.fail('三角定位拍照前模组坐标未停稳', code='TOOLOFFSET-SETTLE', where='90 坐标稳定检查', hint='检查模组振动、PLC 到位时序与 jitter_tol_mm。')
                self.actual = actual
                if s['tri_session'] is not None and s['tri_session'].target_pose is not None:
                    expected = s['tri_session'].target_pose
                    tol = ctx.config.tool_offset['triangulation']['position_tolerance_mm']
                    if any(abs(a - b) > tol for a, b in zip(actual, expected)):
                        return self.fail(f'相机到位超差：期望 {expected}，实际 {actual}，公差 {tol} mm', code='TOOLOFFSET-ARRIVAL', where='90 相机到位核对', hint='检查 PLC 运动控制与到位公差。')
                self.tri_frames = []
                self.tri_target_frames = ctx.config.tool_offset['triangulation']['frames_per_shot']
                self.start_call('camera', {'stage': 'tooloffset', 'task_id': bb.read('task_id')}, 'tooloffset_image')
                self.stage = 'capture_triangulate'
                return Status.RUNNING

            if self.stage == 'capture_triangulate':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return Status.RUNNING
                if status == Status.FAILURE:
                    return self.fail('主相机采图失败：' + str(bb.read('error')), code='CAMERA-CAPTURE', where='90 主相机采图', hint='检查主相机连接与触发。')
                self.tri_frames.append(bb.read('tooloffset_image'))
                if len(self.tri_frames) < self.tri_target_frames:
                    self.start_call('camera', {'stage': 'tooloffset', 'task_id': bb.read('task_id')}, 'tooloffset_image')
                    return Status.RUNNING
                return self._handle_triangulate()

            if self.stage == 'settle_tool':
                actual = self.settle()
                if actual is None:
                    return Status.RUNNING
                if actual is False:
                    return self.fail('对刀拍照前模组坐标未停稳', code='TOOLOFFSET-SETTLE', where='100/110 坐标稳定检查', hint='检查模组振动、PLC 到位时序与 jitter_tol_mm。')
                self.actual = actual
                align_sess = s.get('align_session')
                if align_sess is not None and align_sess.last_target is not None:
                    tol = ctx.config.tool_offset['align']['position_tolerance_mm']
                    if any(abs(a - b) > tol for a, b in zip(actual, align_sess.last_target)):
                        return self.fail(f'工具到位超差：期望 {align_sess.last_target}，实际 {actual}，公差 {tol} mm', code='TOOLOFFSET-ARRIVAL', where='100/110 工具到位核对', hint='检查 PLC 运动控制与到位公差。')
                from nest_clear.baseline.runout import FRAME_COUNT
                self.start_call('calib_camera', {
                    'task_id': bb.read('task_id'),
                    'request_func': bb.read('request_func'),
                    'actual': actual,
                    'target': align_sess.last_target if align_sess and align_sess.last_target else actual,
                    'iteration': s['shots'] + 1,
                    'frames_per_view': FRAME_COUNT,
                    'capture_after': time.monotonic(),
                }, 'tooloffset_calib_images')
                self.stage = 'capture_tool'
                return Status.RUNNING

            if self.stage == 'capture_tool':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return Status.RUNNING
                if status == Status.FAILURE:
                    return self.fail('对刀相机采图失败：' + str(bb.read('error')), code='CALIB-CAMERA-CAPTURE', where='100/110 对刀相机采图', hint='检查正面/侧面相机连接与触发。')
                return self._handle_tool_align()

            if self.stage == 'verify_130':
                status = self.tick_call()
                if status == Status.RUNNING:
                    return Status.RUNNING
                if status == Status.FAILURE:
                    return self.fail('读取 PLC 130 最终坐标失败：' + str(bb.read('error')), code='COMM-READ', where='130 坐标读取', hint='检查 PLC 连接与寄存器 40917–40927。')
                return self._handle_verify_130()

            raise ValueError(f'未知工具偏移阶段 {self.stage}')
        except GuardError as exc:
            return self.fail(exc.what, code=exc.code, where=exc.where, hint=exc.hint)
        except Exception as exc:
            return self.fail(str(exc))

    def _handle_triangulate(self):
        raise NotImplementedError('三角定位将在 T86-2 实现')

    def _handle_tool_align(self):
        raise NotImplementedError('对刀相机对准将在 T86-3 实现')

    def _handle_verify_130(self):
        raise NotImplementedError('130 核对保存将在 T86-3 实现')

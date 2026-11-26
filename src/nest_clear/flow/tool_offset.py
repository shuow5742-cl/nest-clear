"""PLC 90/95 工具偏移标定；所有等待均逐拍轮询。"""
from datetime import datetime
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
from nest_clear.tooloffset.session import Session, stats
from nest_clear.tooloffset.store import load_offset, save_confirmed
from .calibration import change_mode
from .leaves import CallWorker
from .modes import Mode


def reader(ctx):
    if not hasattr(ctx.events, 'command_reader'):
        ctx.events.command_reader = CommandReader(ctx.events.directory, ctx.events)
    return ctx.events.command_reader


def finish(ctx, bb, outcome, reason=None):
    session = bb.read('tooloffset_session', None)
    if session:
        engine = session.get('engine')
        record = {'tool': session.get('tool'), 'mode': session.get('mode'), 'start_time': session['start_time'],
                  'end_time': ctx.events.clock().isoformat(), 'outcome': outcome, 'reason': reason,
                  'samples': engine.samples if engine else [], 'edge_validation': engine.edges if engine else [],
                  'skipped': engine.skipped if engine else [], 'commands': session['commands'],
                  'camera_calibration_sha256': session.get('hash'), 'requests': session['requests'], 'shots':session['shots']}
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
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not isinstance(data, dict): data = {}
    except (OSError, ValueError, yaml.YAMLError):
        data = {}
    return {f'tool{i}': {**(data.get(f'tool{i}') or {}),
                         'calibration_changed': bool(data.get(f'tool{i}') and data[f'tool{i}'].get('camera_calibration_sha256') != digest),
                         'current_tools_xy': list(ctx.config.tools[i-1][:2])} for i in (1, 2)}


class ToolOffset(TreeNode):
    def __init__(self, ctx):
        super().__init__('工具偏移标定')
        self.ctx = ctx
        self.call = None

    def publish(self, phase, message='', **extra):
        s = self._blackboard.read('tooloffset_session', None)
        engine = s.get('engine') if s else None
        fields = dict(phase=phase, message=message, tool=s.get('tool') if s else None,
                      mode=s.get('mode') if s else None, command_seq=reader(self.ctx).last_seq)
        if s:
            fields['existing'] = _existing(self.ctx, s['path'], s['hash'])
            fields['recognition_warning'] = s.get('recognition_warning', '')
        if engine:
            mean, std, se = stats(engine.samples)
            display_samples = [dict(sample, distance_from_mean_mm=float(np.linalg.norm(np.asarray(sample['offset_xy_mm']) - mean)) if mean is not None else 0.) for sample in engine.samples]
            fields.update(target=f'T{engine.current:02d}' if engine.current is not None else None,
                          samples=display_samples, edges=engine.edges, skipped=engine.skipped,
                          mean=mean.tolist() if mean is not None else None,
                          std=std.tolist() if std is not None and np.isfinite(std).all() else None,
                          se=se.tolist() if se is not None and np.isfinite(se).all() else None,
                          se_tol_mm=engine.cfg['se_tol_mm'], min_points=engine.cfg['min_points'], edge_warn_mm=engine.cfg['edge_warn_mm'])
        fields.update(extra)
        self.ctx.events.emit('tooloffset_state', **fields)

    def set_reply(self, code, coords=None, reason=None):
        bb = self._blackboard
        bb.write('reply_func', code, self.name)
        bb.write('reply_coordinates', coords or {}, self.name)
        bb.write('reply_tool', None, self.name)
        if reason:
            bb.write('reply_diagnostic', diagnostic(reason, code='TOOLOFFSET-FAILED', where='工具偏移标定', hint='检查标定卡、位置和光照。'), self.name)
        return Status.SUCCESS

    def fail(self, reason, business=True):
        self.publish('failed', reason)
        finish(self.ctx, self._blackboard, 'failed', reason)
        if business:
            return self.set_reply(92, reason=reason)
        self._blackboard.write('error', reason, self.name)
        self._blackboard.write('error_diagnostic', diagnostic(reason, code='TOOLOFFSET-PROTOCOL', where='工具偏移标定'), self.name)
        return Status.FAILURE

    def on_start(self):
        self.call = None
        bb, ctx = self._blackboard, self.ctx
        s = bb.read('tooloffset_session', None)
        func = bb.read('request_func')
        try:
            if func == 90 and s is None:
                start = ctx.config.camera_calibration.get('start_xyz')
                if start is None:
                    raise GuardError('CAMCAL-START-UNSET', '未配置相机标定起始坐标', '工具偏移标定', '配置 camera_calibration.start_xyz。')
                path = Path(ctx.calibration_path or ctx.config.vision.calibration)
                if ctx.calibration is None or getattr(ctx.calibration, 'placeholder', False) or not path.is_file():
                    raise ValueError('相机标定文件不存在或未完成标定')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                travel = ctx.config.travel
                if any(not lo <= v <= hi for v, (lo, hi) in zip(start, travel)):
                    raise ValueError('工具偏移标定起始坐标超出行程')
                now = ctx.events.clock()
                s = {'phase':'select', 'tool':None, 'mode':None, 'engine':None, 'start':tuple(start), 'travel':travel,
                     'path':path.parent / 'tool_offset.yaml', 'hash':digest, 'shots':0, 'commands':[],
                     'start_time':now.isoformat(), 'directory':now.strftime('%Y%m%d_%H%M%S_%f'), 'requests':[{'task_id':bb.read('task_id'), 'func':90}]}
                bb.write('tooloffset_session', s, self.name)
                change_mode(bb, ctx, Mode.TOOL_OFFSET, self.name)
                ctx.events.abort_tooloffset = lambda reason: finish(ctx, bb, 'aborted', reason)
                self.stage = 'select'
                self.publish('select', '请选择工具和模式并开始')
                return Status.RUNNING
            if s is None:
                raise ValueError('无工具偏移会话时收到 95')
            s['requests'].append({'task_id':bb.read('task_id'), 'func':func})
            if func == 90:
                if s['phase'] != 'wait90':
                    return self.fail('工具偏移会话中收到错序 90', False)
                s['phase'] = 'wait95'
                self.publish('moving', '边缘验证中，不需要移动工具' if s['engine'].phase == 'edge' else '相机移动中')
                return self.set_reply(90, {'scan_xyz': tuple(s['engine'].target_pose)})
            if func != 95 or s['phase'] != 'wait95':
                return self.fail('工具偏移会话收到错序功能码', False)
            self.stage = 'settle_arrival'
            self.samples, self.round, self.next_read = [], 0, 0.
            self.publish('arrival', '边缘验证中，不需要移动工具' if s['engine'].phase == 'edge' else '检查相机到位')
            return Status.RUNNING
        except Exception as exc:
            if bb.read('tooloffset_session', None):
                finish(ctx, bb, 'aborted', str(exc))
            bb.write('error', str(exc), self.name)
            bb.write('error_diagnostic', diagnostic(exc, where=self.name), self.name)
            return Status.FAILURE

    def command(self, allowed):
        item = reader(self.ctx).poll(set(allowed) | {'abort'})
        if item:
            s = self._blackboard.read('tooloffset_session')
            s['commands'].append({'seq':reader(self.ctx).last_seq, 'command':item[0], 'params':item[1]})
            self.publish(self.stage)
        return item

    def start_call(self, target, payload, result):
        self.call = CallWorker(self.ctx.board, target, payload, result, name='工具偏移设备调用')
        self.call.set_blackboard(self._blackboard)

    def tick_call(self):
        status = self.call.tick(self.snapshot, self.tick_ctx)
        if status != Status.RUNNING:
            self.call = None
        return status

    def settle(self):
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
            self.start_call('plc', {'operation':'actual'}, 'tooloffset_actual')
        return None

    def process_result(self, result):
        s = self._blackboard.read('tooloffset_session')
        engine = s['engine']
        if result == 'failed':
            return self.fail(engine.failure)
        if result == 'continue':
            s['phase'] = 'wait90'
            message = '边缘验证中，不需要移动工具' if engine.phase == 'edge' else '等待下一相机位置'
            if engine.samples and engine.samples[-1]['outlier'] and engine.samples[-1]['target'] not in s.get('reported_outliers', []):
                sample = engine.samples[-1]; message = f"第 {len(engine.samples)} 点（{sample['target']}）离群，已剔除并加入重测"
                s.setdefault('reported_outliers', []).append(sample['target'])
            self.publish('wait90', message)
            return self.set_reply(95)
        if result == 'tool':
            self.stage = 'tool'
            self.publish('tool', f'请手动移动工具{s["tool"]}到 T{engine.current:02d} 中心，到位后按【工具到位】')
            return Status.RUNNING
        if result == 'confirm':
            self.stage = 'confirm'
            self.publish('confirm', '等待确认覆盖', candidate=engine.result())
            return Status.RUNNING
        raise ValueError(f'未知工具偏移阶段 {result}')

    def on_running(self):
        bb, ctx = self._blackboard, self.ctx
        s = bb.read('tooloffset_session', None)
        if s is None:
            return self.fail('工具偏移会话已结束', False)
        try:
            allowed = {'select': {'select_tool_and_start'}, 'tool': {'tool_in_place'}, 'confirm': {'confirm_overwrite'}}.get(self.stage, set())
            cmd = self.command(allowed)
            if cmd and cmd[0] == 'abort':
                return self.fail('用户放弃')
            if self.stage == 'select':
                if not cmd: return Status.RUNNING
                tool, mode = cmd[1].get('tool'), cmd[1].get('mode')
                if type(tool) is not int or tool not in (1,2) or mode not in ('center','center_edge'):
                    self.publish('select', '工具或模式无效')
                    return Status.RUNNING
                try:
                    prior = load_offset(s['path'], tool)
                except (ValueError, OSError, yaml.YAMLError):
                    prior = None
                s.update(tool=tool, mode=mode, engine=Session(ctx.calibration, s['start'], s['travel'], ctx.config.tool_offset, mode, tool, board_thickness_mm=ctx.config.camera_calibration['board_thickness_mm'], prior_offset=prior), phase='wait95')
                self.publish('moving', '相机移动至起始位置')
                return self.set_reply(90, {'scan_xyz':s['start']})
            if self.stage.startswith('settle_'):
                actual = self.settle()
                if actual is None: return Status.RUNNING
                if actual is False:
                    if self.stage == 'settle_tool':
                        self.stage = 'tool'
                        self.publish('tool', '轴还在动或抖动过大，请稍后再按【工具到位】')
                        return Status.RUNNING
                    return self.fail('相机到位坐标未停稳', False)
                if self.stage == 'settle_tool':
                    return self.process_result(s['engine'].add_sample(actual))
                expected = s['engine'].target_pose
                tolerance = ctx.config.camera_calibration.get('position_tolerance_mm', .05)
                if any(abs(a-b) > tolerance for a,b in zip(actual, expected)):
                    return self.fail(f'相机到位超差：期望 {expected}，实际 {actual}', False)
                self.actual = actual
                self.start_call('camera', {'stage':'tooloffset', 'task_id':bb.read('task_id')}, 'tooloffset_image')
                self.stage = 'capture'
                return Status.RUNNING
            if self.stage == 'capture':
                status = self.tick_call()
                if status == Status.RUNNING: return status
                if status == Status.FAILURE: return self.fail('采图失败：' + str(bb.read('error')))
                self.start_call('tooloffset', {'captured':bb.read('tooloffset_image'), 'context':ctx, 'session':s, 'task_id':bb.read('task_id')}, 'tooloffset_detection')
                self.stage = 'detect'
                return Status.RUNNING
            if self.stage == 'detect':
                status = self.tick_call()
                if status == Status.RUNNING: return status
                if status == Status.FAILURE: return self.fail('目标识别失败：' + str(bb.read('error')))
                data = bb.read('tooloffset_detection')
                current = next((t for t in data['targets'] if t.index == s['engine'].current), None)
                s['recognition_warning'] = (f'{current.label} 圆环与十字中心差 {current.cross_error_mm:.3f} mm，请检查打印和光照'
                                            if current and current.cross_error_mm > s['engine'].cfg['ring_cross_tol_mm'] else '')
                if s['recognition_warning']:
                    ctx.events.emit('tooloffset_quality_warning', target=current.label, cross_error_mm=current.cross_error_mm, reason='圆环与十字中心差异偏大')
                bb.write('raw_path', data['raw_path'], self.name)
                return self.process_result(s['engine'].image(data['targets'], self.actual, data['size']))
            if self.stage == 'tool':
                if not cmd: return Status.RUNNING
                self.stage, self.samples, self.round, self.next_read = 'settle_tool', [], 0, 0.
                return Status.RUNNING
            if self.stage == 'confirm':
                if not cmd: return Status.RUNNING
                accept = cmd[1].get('accept')
                if type(accept) is not bool:
                    self.publish('confirm', '确认参数无效', candidate=s['engine'].result())
                    return Status.RUNNING
                if not accept: return self.fail('用户取消保存')
                result = s['engine'].result()
                try:
                    commit = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True, timeout=2, check=True).stdout.strip()
                except (OSError, subprocess.SubprocessError): commit = None
                result.update(time=ctx.events.clock().isoformat(), camera_calibration_sha256=s['hash'], git_commit=commit)
                _, _, xyz = save_confirmed(s['path'], Path(ctx.config.tool_calibration.directory) / 'tools.yaml', s['tool'], result, task_id=bb.read('task_id'))
                offsets = list(ctx.config.tools); offsets[s['tool']-1] = xyz
                object.__setattr__(ctx.config, 'tools', tuple(offsets))
                from .pick_accuracy import refresh_existing
                refresh_existing(ctx)
                from nest_clear.baseline.session import refresh_existing as refresh_baseline_existing
                refresh_baseline_existing(ctx)
                self.publish('done', '保存成功', candidate=result)
                finish(ctx, bb, 'done')
                return self.set_reply(91)
            raise ValueError('未知等待阶段')
        except Exception as exc:
            return self.fail(str(exc), business=self.stage not in ('settle_arrival',))

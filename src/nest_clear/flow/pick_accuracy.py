"""PLC 150/155/156 挑取精度标定。"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from autoweaver.motion_policy.nodes.node import Status
from nest_clear.guard import GuardError, diagnostic
from nest_clear.pickaccuracy.session import Session
from nest_clear.pickaccuracy.store import has_calibrated_z, read, save
from .calibration import change_mode
from .modes import Mode
from .picking import calculate_pick_coords
from .tool_offset import ToolOffset, reader


def offset_digest(ctx, tool):
    """对单把工具的偏移结果作稳定快照，另一把工具更新不影响它。"""
    import yaml
    path=Path(ctx.config.tool_calibration.directory)/'tool_offset.yaml'
    try:
        data=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        result=data.get(f'tool{tool}')
        if not isinstance(result,dict): return None
        encoded=json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return None


def finish(ctx, bb, outcome, reason=None):
    s=bb.read('pickaccuracy_session',None)
    if s:
        engine=s.get('engine')
        record={'tool':s.get('tool'), 'start_time':s['start_time'], 'end_time':ctx.events.clock().isoformat(),
                'outcome':outcome,'reason':reason,'requests':s['requests'], 'commands':s['commands'],
                'shots':s['shots'], 'result':engine.result() if engine and engine.candidate else None,
                'samples':engine.samples if engine else [], 'skipped':engine.skipped if engine else [],
                'plan':engine.plans if engine else {}}
        if ctx.events.run_directory is not None:
            path=ctx.events.run_directory/'pickacc'/s['directory']/'session_record.json'
            try:
                path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            except (OSError,ValueError,TypeError) as exc: ctx.events.emit('error',operation='pickacc_record',reason=str(exc))
        ctx.events.emit('pickacc_'+outcome,**record)
    if bb.read('mode') == Mode.PICK_ACCURACY: change_mode(bb,ctx,Mode.IDLE,'结束挑取精度标定')
    bb.write('pickaccuracy_session',None,'结束挑取精度标定'); ctx.events.abort_pickaccuracy=None


def existing(ctx,path,digest):
    data=read(path); result={}
    for i in (1,2):
        old=data.get(f'tool{i}') or {}
        result[f'tool{i}']={**old,
                            'z_calibrated': has_calibrated_z(old),
                            'stale':bool(old and (old.get('camera_calibration_sha256') != digest or old.get('tool_offset_xy') != list(ctx.config.tools[i-1][:2])))}
    return result


def _current_tool_z_ref(ctx, tool):
    """读取保存挑取精度时 tools.yaml 中该工具的 offset_xyz[2] 作为 Z_ref。"""
    tools_path = Path(ctx.config.tool_calibration.directory) / 'tools.yaml'
    z_ref = float(ctx.config.tools[tool - 1][2])
    if tools_path.is_file():
        try:
            from nest_clear.tooltip.store import load_tools
            item = load_tools(tools_path)['tools'][f'tool{tool}']
            if item.get('source') != '占位，未标定' or tuple(ctx.config.tools[tool - 1]) == (0.0, 0.0, 0.0):
                z_ref = float(item['offset_xyz'][2])
        except (GuardError, KeyError, TypeError, ValueError):
            pass
    return z_ref


def refresh_existing(ctx):
    """其他标定完成后刷新失效提示，不能影响原标定协议。"""
    try:
        camera = Path(ctx.calibration_path or ctx.config.vision.calibration)
        digest = hashlib.sha256(camera.read_bytes()).hexdigest()
        path = Path(ctx.config.tool_calibration.directory) / 'pick_compensation.yaml'
        ctx.events.emit('pickacc_state', existing=existing(ctx, path, digest))
    except Exception as exc:
        ctx.events.emit('warning', reason=f'刷新挑取补偿状态失败：{exc}')


class PickAccuracy(ToolOffset):
    def __init__(self,ctx):
        super().__init__(ctx); self.name='挑取精度标定'; self.z_jitter_mm=None

    def publish(self,phase,message='',**extra):
        s=self._blackboard.read('pickaccuracy_session',None); e=s.get('engine') if s else None
        fields={'phase':phase,'message':message,'tool':s.get('tool') if s else self._blackboard.read('plc_tool_id'),
                'plc_tool_id':self._blackboard.read('plc_tool_id'),'command_seq':reader(self.ctx).last_seq,
                'z_jitter_mm':getattr(self,'z_jitter_mm',None)}
        if s: fields['existing']=existing(self.ctx,s['path'],s['hash'])
        if e:
            z_points=[{'target':item['target'],'stage':item['stage'],'pick_z_mm':item['pick_z_mm']}
                      for item in e.samples if not item.get('discarded') and item.get('pick_z_mm') is not None]
            fields.update(target=f'T{e.current:02d}' if e.current is not None else None,
                          samples=e.samples, skipped=e.skipped, infield=e.edges, warnings=e.warnings,
                          decision=e.decision, candidate=e.candidate, plan=e.plans, predicted={f'T{k:02d}':v.tolist() for k,v in e.predicted.items()},
                          queue=[{'target':f'T{i:02d}','stage':st} for i,st,_ in e.queue],
                          pending_sample=e.pending_sample, z_points=z_points, z_summary=e.z_summary(),
                          last_z_mm=z_points[-1]['pick_z_mm'] if z_points else None)
        fields.update(extra); self.ctx.events.emit('pickacc_state',**fields)

    def settle_z(self, cfg):
        """采样 6 次/0.5s 实时坐标，Z 极差 <= 0.02mm 时返回 (mean_z, z_range)；否则返回 (False, z_range)。"""
        if self.call:
            status = self.tick_call()
            if status == Status.FAILURE:
                raise ValueError('读取实时坐标失败：' + str(self._blackboard.read('error')))
            if status == Status.RUNNING:
                return None, None
            value = self._blackboard.read('tooloffset_actual')
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError('实时坐标无效')
            self.samples.append(value)
            n = int(cfg['z_settle_samples'])
            self.next_read = time.monotonic() + float(cfg['z_settle_window_s']) / max(1, n - 1)
        n = int(cfg['z_settle_samples'])
        if len(self.samples) >= n:
            zs = np.asarray([sample[2] for sample in self.samples], dtype=float)
            z_range = float(np.ptp(zs))
            self.z_jitter_mm = z_range
            if z_range <= float(cfg['z_jitter_tol_mm']):
                return float(zs.mean()), z_range
            return False, z_range
        if self.call is None and time.monotonic() >= self.next_read:
            self.start_call('plc', {'operation': 'actual'}, 'tooloffset_actual')
        return None, None

    def command(self,allowed):
        item=reader(self.ctx).poll(set(allowed)|{'abort'})
        if item:
            s=self._blackboard.read('pickaccuracy_session'); s['commands'].append({'seq':reader(self.ctx).last_seq,'command':item[0],'params':item[1]})
        return item

    def set_reply(self,code,coords=None,reason=None):
        bb=self._blackboard; bb.write('reply_func',code,self.name); bb.write('reply_coordinates',coords or {},self.name); bb.write('reply_tool',None,self.name)
        if reason: bb.write('reply_diagnostic',diagnostic(reason,code='PICKACC-FAILED',where=self.name),self.name)
        return Status.SUCCESS

    def fail(self,reason,business=True):
        self.publish('failed',reason); finish(self.ctx,self._blackboard,'failed',reason)
        if business: return self.set_reply(152,reason=reason)
        self._blackboard.write('error',reason,self.name); self._blackboard.write('error_diagnostic',diagnostic(reason,code='PICKACC-PROTOCOL',where=self.name),self.name)
        return Status.FAILURE

    def on_start(self):
        self.call=None; bb=self._blackboard; ctx=self.ctx; s=bb.read('pickaccuracy_session',None); func=bb.read('request_func')
        try:
            if func==150 and s is None:
                start=ctx.config.camera_calibration.get('start_xyz'); path=Path(ctx.calibration_path or ctx.config.vision.calibration)
                if start is None: raise ValueError('未配置相机标定起始坐标')
                if ctx.calibration is None or getattr(ctx.calibration,'placeholder',False) or not path.is_file(): raise ValueError('相机标定文件不存在或未完成标定')
                configured=ctx.config.travel
                travel=tuple((max(lo,safe_lo),min(hi,safe_hi)) for (lo,hi),(safe_lo,safe_hi) in zip(configured,((80,180),(-10,150),(-15,0))))
                if any(lo>hi for lo,hi in travel): raise ValueError('配置行程与挑取精度标定安全行程不相交')
                if any(not lo<=v<=hi for v,(lo,hi) in zip(start,travel)): raise ValueError('起始坐标超出行程')
                now=ctx.events.clock(); s=dict(phase='select',tool=bb.read('plc_tool_id'),engine=None,start=tuple(start),travel=travel,
                    path=Path(ctx.config.tool_calibration.directory)/'pick_compensation.yaml',hash=hashlib.sha256(path.read_bytes()).hexdigest(),shots=0,commands=[],
                    start_time=now.isoformat(),directory=now.strftime('%Y%m%d_%H%M%S_%f'),requests=[{'task_id':bb.read('task_id'),'func':func}],kind='pickacc')
                bb.write('pickaccuracy_session',s,self.name); change_mode(bb,ctx,Mode.PICK_ACCURACY,self.name)
                ctx.events.abort_pickaccuracy=lambda reason: finish(ctx,bb,'aborted',reason)
                self.stage='select'; self.publish('select','请按【开始】'); return Status.RUNNING
            if s is None: raise ValueError('无挑取精度会话')
            s['requests'].append({'task_id':bb.read('task_id'),'func':func})
            if func==150 and s['phase']=='wait150':
                s['phase']='wait155'; self.publish('moving','相机移动中'); return self.set_reply(150,{'scan_xyz':tuple(s['engine'].target_pose)})
            if func==155 and s['phase']=='wait155':
                self.stage='settle_arrival'; self.samples=[]; self.round=0; self.next_read=0.; self.publish('arrival','检查相机到位'); return Status.RUNNING
            if func==156 and s['phase']=='wait156':
                self.stage='tool'; self.z_jitter_mm=None; self.publish('tool_in_place','工具已到位，调整好后按【计算补偿】'); return Status.RUNNING
            return self.fail('挑取精度标定收到错序功能码',False)
        except Exception as exc:
            if bb.read('pickaccuracy_session',None): finish(ctx,bb,'aborted',str(exc))
            bb.write('error',str(exc),self.name); bb.write('error_diagnostic',diagnostic(exc,where=self.name),self.name); return Status.FAILURE

    def process_result(self,result):
        s=self._blackboard.read('pickaccuracy_session'); e=s['engine']
        if result=='failed': return self.fail(e.failure)
        if result=='continue':
            s['phase']='wait155'; self.publish('moving','相机移动中'); return self.set_reply(150,{'scan_xyz':tuple(e.target_pose)})
        if result=='tool':
            target=e.measured[e.current]; comp=e._verification_comp(e.current,target) if e.plan[0]=='verify' else (0.,0.)
            # Z 为标定配置绝对值，不叠加工具 Z。
            offsets=[(o[0],o[1],0.) for o in self.ctx.config.tools]
            try:
                xyz=calculate_pick_coords((*target,0.),offsets,e.cfg['calib_tool_z_mm'],s['travel'],tool=s['tool'],compensation=lambda _t,_xy:comp)
            except Exception as exc:
                e.skipped.append({'target':f'T{e.current:02d}', 'stage':e.plan[0], 'reason':f'识别后工具坐标不可达：{exc}'})
                return self.process_result(e._move_next())
            e.sent=xyz; s['phase']='wait156'; self.publish('wait_tool','请等待工具前往'); return self.set_reply(155,{f'calibration{s["tool"]}_xyz':xyz})
        if result=='confirm':
            self.stage='confirm'
            candidate=e.result(); candidate['z_ref']=_current_tool_z_ref(self.ctx,s['tool'])
            self.publish('confirm','等待确认覆盖',candidate=candidate); return Status.RUNNING
        raise ValueError('未知挑取精度阶段')

    def on_running(self):
        bb=self._blackboard; ctx=self.ctx; s=bb.read('pickaccuracy_session',None)
        if s is None: return self.fail('标定会话已结束',False)
        try:
            allowed={'select':{'pickacc_start'},'tool':{'pickacc_calculate','pickacc_discard'},'z_step':{'pickacc_record_z','pickacc_discard'},'confirm':{'pickacc_confirm'}}.get(self.stage,set())
            cmd=self.command(allowed)
            if cmd and cmd[0]=='abort': return self.fail('用户放弃')
            if self.stage=='select':
                if not cmd: return Status.RUNNING
                tool=s['tool']
                prior=read(s['path']).get('tool1') if tool==2 else None
                current_offset_hash=offset_digest(ctx,1)
                if prior and (prior.get('camera_calibration_sha256')!=s['hash'] or prior.get('tool_offset_xy')!=list(ctx.config.tools[0][:2]) or not current_offset_hash or prior.get('tool_offset_result_sha256')!=current_offset_hash): prior=None
                cfg={**ctx.config.tool_offset,**ctx.config.pick_accuracy}
                s.update(engine=Session(ctx.calibration,s['start'],s['travel'],cfg,tool,ctx.config.tools,ctx.config.vision.fine_box_half_mm,ctx.config.camera_calibration['board_thickness_mm'],prior),phase='wait155')
                self.publish('moving','相机移动至起始位置'); return self.set_reply(150,{'scan_xyz':s['start']})
            if self.stage=='settle_z':
                e=s['engine']
                mean_z,z_range=self.settle_z(e.cfg)
                if mean_z is None: return Status.RUNNING
                if mean_z is False:
                    self.stage='z_step'
                    self.publish('z_step',f'Z 轴未停稳（当前极差 {z_range:.3f} mm > {e.cfg["z_jitter_tol_mm"]:.3f} mm），请等待停稳后再按【记录下针 Z】')
                    return Status.RUNNING
                try:
                    result=e.record_z(mean_z,z_jitter_mm=z_range)
                except GuardError as exc:
                    if exc.code=='POSITION-TRAVEL':
                        self.stage='z_step'; self.samples=[]
                        ctx.events.emit('warning',reason=str(exc),target=f'T{e.current:02d}')
                        self.publish('z_step',f'下针 Z {mean_z:.3f} mm 超出 Z 行程 {s["travel"][2]}，请调整后重新按【记录下针 Z】')
                        return Status.RUNNING
                    raise
                if result=='continue':
                    s['phase']='wait150'
                    last=e.samples[-1]
                    message='疑似对准失误，已排入重测' if last.get('outlier') else '等待下一相机位置'
                    self.publish('wait150',message)
                    if last.get('outlier'): ctx.events.emit('warning',reason=message,target=last['target'])
                    return self.set_reply(156)
                return self.process_result(result)
            if self.stage.startswith('settle_'):
                actual=self.settle()
                if actual is None: return Status.RUNNING
                if actual is False:
                    if self.stage=='settle_tool': self.stage='tool'; self.publish('tool_in_place','轴还在动或抖动过大，请稍后再按【计算补偿】'); return Status.RUNNING
                    return self.fail('相机到位坐标未停稳',False)
                if self.stage=='settle_tool':
                    s['engine'].add_xy_sample(actual)
                    self.stage='z_step'; self.z_jitter_mm=None
                    self.publish('z_step','XY 已记录；请放 0.05mm 塞尺并点动工具尖刚好压住塞尺，按【记录下针 Z】')
                    return Status.RUNNING
                expected=s['engine'].target_pose; tolerance=ctx.config.camera_calibration.get('position_tolerance_mm',.05)
                if any(abs(a-b)>tolerance for a,b in zip(actual,expected)): return self.fail(f'相机到位超差：期望 {expected}，实际 {actual}',False)
                self.actual=actual; s['engine'].capture_pose=actual
                self.start_call('camera',{'stage':'pickacc','task_id':bb.read('task_id')},'tooloffset_image'); self.stage='capture'; return Status.RUNNING
            if self.stage=='capture':
                status=self.tick_call()
                if status==Status.RUNNING: return status
                if status==Status.FAILURE: return self.fail('采图失败')
                self.start_call('tooloffset',{'captured':bb.read('tooloffset_image'),'context':ctx,'session':s,'task_id':bb.read('task_id'),'pose':self.actual},'tooloffset_detection'); self.stage='detect'; return Status.RUNNING
            if self.stage=='detect':
                status=self.tick_call()
                if status==Status.RUNNING: return status
                if status==Status.FAILURE: return self.fail('目标识别失败')
                data=bb.read('tooloffset_detection'); return self.process_result(s['engine'].image(data['targets'],self.actual,data['size']))
            if self.stage in ('tool','z_step'):
                if not cmd: return Status.RUNNING
                if cmd[0]=='pickacc_discard':
                    result=s['engine'].discard()
                    if result=='failed': return self.fail(s['engine'].failure)
                    s['phase']='wait150'; self.z_jitter_mm=None; self.publish('wait150','本点作废，等待重测'); return self.set_reply(156)
                if self.stage=='tool':
                    self.stage='settle_tool'; self.samples=[]; self.round=0; self.next_read=0.; return Status.RUNNING
                self.stage='settle_z'; self.samples=[]; self.next_read=0.; return Status.RUNNING
            if self.stage=='confirm':
                if not cmd: return Status.RUNNING
                accept=cmd[1].get('accept')
                if type(accept) is not bool: return Status.RUNNING
                if not accept: return self.fail('用户取消保存')
                try: commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=2,check=True).stdout.strip()
                except (OSError,subprocess.SubprocessError): commit=None
                result=s['engine'].result(); result.update(time=ctx.events.clock().isoformat(),git_commit=commit,camera_calibration_sha256=s['hash'],tool_offset_xy=list(ctx.config.tools[s['tool']-1][:2]),tool_offset_result_sha256=offset_digest(ctx,s['tool']),z_ref=_current_tool_z_ref(ctx,s['tool']))
                save(s['path'],s['tool'],result); self.publish('done','保存成功',candidate=result); finish(ctx,bb,'done'); return self.set_reply(151)
            raise ValueError('未知等待阶段')
        except Exception as exc: return self.fail(str(exc),business=self.stage not in ('settle_arrival','settle_z'))

from dataclasses import replace
import numpy as np
import pytest
from nest_clear.pickaccuracy.session import Session
from nest_clear.pickaccuracy.store import evaluate, save, read
from nest_clear.vision.geometry import Calibration
from test_tooloffset_flow import rig


def engine(tmp_path, tool=1, prior=None, **cfg):
    cal=Calibration(np.array([[2000.,0,2000],[0,2000.,1500],[0,0,1]]),np.zeros(5),np.diag([1.,-1.,-1.]),np.zeros(3),-105.)
    e=Session(cal,(130.,70.,-5.),((80,180),(-10,150),(-15,0)),cfg,tool,((0,0,0),(0,0,0)),4.,prior=prior)
    e.image_size=(4000,3000)
    e.predicted={i:np.asarray([130.,70.])+p for i,p in e.nominal.items()}
    return e


def run(e, model, noise=None):
    result=e._move_next(); n=0
    while result=='continue':
        n+=1; point=e.predicted[e.current]; e.measured[e.current]=point; e.phase='tool'
        comp=e._verification_comp(e.current,point) if e.plan[0]=='verify' else (0,0)
        e.sent=(*tuple(point+comp),-10.)
        error=np.asarray(model(*point))
        if noise: error+=noise(n,e)
        result=e.add_sample((*tuple(np.asarray(e.sent[:2])+error-np.asarray(comp)),-10.))
        assert n<50
    return result


def test_constant_and_table(tmp_path):
    e=engine(tmp_path)
    assert run(e,lambda x,y:(.08,-.05))=='confirm'
    assert e.candidate['type']=='constant'
    assert len(e.edges)==4 and len(e.samples)==15
    assert evaluate(e.candidate,(130,70))==pytest.approx((.08,-.05))
    def model(x,y): return (.002*(x-130)+.000002*(y-70)**2, -.001*(y-70))
    e=engine(tmp_path)
    assert run(e,model)=='confirm', e.failure
    assert e.candidate['type']=='table'
    rng=np.random.default_rng(7)
    for x,y in rng.uniform([85,25],[175,115],size=(200,2)):
        assert np.linalg.norm(np.asarray(model(x,y))-evaluate(e.candidate,(x,y)))<.05
    path=tmp_path/'pick_compensation.yaml'; save(path,1,e.result()); save(path,2,{'type':'constant','constant_xy':[.1,.2]})
    assert read(path)['tool1']['type']=='table'
    assert list((tmp_path/'history').glob('pick_compensation_*.yaml'))


def test_repeat_fail_and_simple(tmp_path):
    def gradient(x,y): return (.002*(x-130),.001*(y-70))
    e=engine(tmp_path)
    assert run(e,gradient,lambda n,e: np.array([.08,0]) if e.plan[0]=='repeat' else np.zeros(2))=='failed'
    assert '不可重复' in e.failure
    base=engine(tmp_path); assert run(base,gradient)=='confirm'
    e=engine(tmp_path,tool=2,prior=base.candidate,min_points_a=5)
    assert run(e,lambda x,y:np.asarray(gradient(x,y))+[.02,-.01])=='confirm'
    assert len(e.samples)==6
    e=engine(tmp_path,tool=2,prior=base.candidate,min_points_a=5,verify_tol_mm=.15)
    assert run(e,lambda x,y:(.004*(x-130),.002*(y-70)))=='confirm'
    assert not e.simple and len(e.samples)>6


def test_wrong_order(rig):
    assert rig.request(1,155)[0]==999
    assert rig.request(2,156)[0]==999
    rig.plc.request=(3,150,1)
    rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    from nest_clear.tooloffset.commands import send
    send(rig.root,'pickacc_start')
    rig.until(lambda:3 in rig.plc.replies)
    assert rig.plc.replies[3][0]==150
    assert rig.request(4,156)[0]==999


@pytest.mark.parametrize('accept', [True, False])
def test_plc_normal_cycle(rig, accept):
    from nest_clear.tooloffset.commands import send
    rig.plc.request=(1,150,1); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    send(rig.root,'pickacc_start'); rig.until(lambda:1 in rig.plc.replies)
    task=1; assert rig.plc.replies[task][0]==150
    assert 'scan_xyz' in rig.plc.replies[task][1] and rig.plc.replies[task][1]['pc_tool_id'] == 1
    for _ in range(80):
        task+=1; rig.plc.request=(task,155,1); rig.until(lambda:task in rig.plc.replies, timeout=35)
        code,coords=rig.plc.replies[task]
        if code==150:
            rig.plc.actual=coords['scan_xyz']; continue
        assert code==155, (code,rig.events.status.data['pickacc'])
        assert list(coords)==['calibration1_xyz', 'pc_tool_id'] and coords['pc_tool_id']==1 and coords['calibration1_xyz'][2]==-10
        xyz=coords['calibration1_xyz']; rig.plc.actual=(xyz[0]+.08,xyz[1]-.05,xyz[2])
        e=rig.batch.blackboard.read('pickaccuracy_session')['engine']
        if e.plan[0]=='verify': rig.plc.actual=xyz
        task+=1; rig.plc.request=(task,156,1)
        rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='tool_in_place')
        assert task not in rig.plc.replies
        send(rig.root,'pickacc_calculate')
        rig.until(lambda:task in rig.plc.replies or rig.events.status.data['pickacc'].get('phase')=='confirm', timeout=35)
        if task not in rig.plc.replies:
            send(rig.root,'pickacc_confirm',{'accept':accept}); rig.until(lambda:task in rig.plc.replies)
        code=rig.plc.replies[task][0]
        if code in (151,152): break
        assert code==156, (code,rig.events.status.data['pickacc'])
        task+=1; code,coords=rig.request(task,150); assert code==150; rig.plc.actual=coords['scan_xyz']
    else: pytest.fail('会话未结束')
    assert code==(151 if accept else 152)
    assert all(coords['pc_tool_id'] == 1 for _, coords in rig.plc.replies.values())
    assert list(rig.events.run_directory.glob('pickacc/*/*_raw.png'))
    if accept: assert read(rig.root/'calibration'/'pick_compensation.yaml')['tool1']['type']=='constant'
    else: assert not (rig.root/'calibration'/'pick_compensation.yaml').exists()


def test_infield_warning_and_maximum(tmp_path):
    e=engine(tmp_path)
    assert run(e,lambda x,y:(.08,-.05),lambda n,e: np.array([.07,0]) if e.plan[0]=='infield' else np.zeros(2))=='confirm'
    assert len(e.edges)==4 and all(edge['warning'] for edge in e.edges)
    assert e.result()['warnings']
    e=engine(tmp_path,max_points=4)
    assert run(e,lambda x,y:(.08,-.05))=='failed'
    assert '最大测点' in e.failure


def test_center_retry(tmp_path):
    from types import SimpleNamespace
    e=engine(tmp_path); assert e._move_next()=='continue'
    pose=e.target_pose; target=SimpleNamespace(index=e.current,ring_uv=(2020.,1500.),cross_error_mm=0.)
    assert e.image([target],pose,(4000,3000))=='continue'
    assert e.target_pose[0] == pytest.approx(pose[0]+1, abs=.002)
    assert e.image([target],e.target_pose,(4000,3000))=='continue'
    assert e.image([target],e.target_pose,(4000,3000))=='failed'
    assert '重试超限' in e.failure


def test_discard_and_abort(rig):
    from nest_clear.tooloffset.commands import send
    rig.plc.request=(1,150,1); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    send(rig.root,'abort'); rig.until(lambda:1 in rig.plc.replies)
    assert rig.plc.replies[1][0]==152
    rig.plc.request=(2,150,2); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    send(rig.root,'pickacc_start'); rig.until(lambda:2 in rig.plc.replies)
    code,coords=rig.request(3,155,2); assert code==150
    rig.plc.actual=coords['scan_xyz']
    code,coords=rig.request(4,155,2); assert code==155 and list(coords)==['calibration2_xyz', 'pc_tool_id'] and coords['pc_tool_id']==2 and coords['calibration2_xyz'][2]==-10
    rig.plc.request=(5,156,2); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='tool_in_place')
    send(rig.root,'pickacc_discard'); rig.until(lambda:5 in rig.plc.replies)
    assert rig.plc.replies[5][0]==156
    e=rig.batch.blackboard.read('pickaccuracy_session')['engine']; assert e.samples[-1]['discarded']
    assert rig.request(6,150,2)[0]==150
    send(rig.root,'abort'); rig.until(lambda:rig.batch.blackboard.read('pickaccuracy_session') is None)


def test_unsupported_start_or_mode(rig):
    object.__setattr__(rig.cfg,'camera_calibration',{**rig.cfg.camera_calibration,'start_xyz':None})
    assert rig.request(1,150)[0]==999
    object.__setattr__(rig.cfg,'camera_calibration',{**rig.cfg.camera_calibration,'start_xyz':[130.,70.,-5.]})
    (rig.root/'calibration'/'camera.yaml').unlink(); assert rig.request(2,150)[0]==999
    from nest_clear.flow.modes import Mode
    rig.batch.blackboard.write('mode',Mode.PICKING,'test'); assert rig.request(3,150)[0]==999


def test_production_compensation_hot_load(tmp_path):
    from nest_clear.config import load_config
    from nest_clear.flow.context import FlowContext, RequestTracker
    from nest_clear.flow.picking import WorkRequest
    from test_hair_integration import Board, Events
    cfg=load_config('config/nest_clear.yaml')
    cfg=replace(cfg,record_directory=str(tmp_path),tools=((1,2,0),(3,4,0)),travel=((-100,100),)*3,tool_calibration=replace(cfg.tool_calibration,directory=str(tmp_path)))
    events=Events(); bb=Board(work_cycle=None,request_func=1,task_id=1)
    request=WorkRequest(FlowContext(None,cfg,[(0,0,0)],RequestTracker(0),events)); request._blackboard=bb; request.perform()
    pick=bb.read('work_cycle')._pick_coords
    assert pick((1,2,0))[0][:2]==(2,4)
    save(tmp_path/'pick_compensation.yaml',1,{'type':'constant','constant_xy':[.08,-.05]})
    assert pick((1,2,0))[0][:2]==pytest.approx((2.08,3.95))
    save(tmp_path/'pick_compensation.yaml',1,{'type':'constant','constant_xy':[.10,-.03]})
    assert pick((1,2,0))[0][:2]==pytest.approx((2.10,3.97))
    assert events.rows[-1][0]=='pick_compensation_used'


def test_selected_register_writes():
    from nest_clear.comm import Handshake, load_contract
    class IO:
        def __init__(self):
            self.words = {}
            self.writes = []
        def write_u16(self,a,v):
            self.writes.append(a)
            self.words[a] = v
        def write_real32_block(self,a,v):
            self.writes.append(a)
            from nest_clear.comm.codec import encode_real32
            for i, word in enumerate(sum((encode_real32(x) for x in v), [])): self.words[a+i]=word
    io=IO(); h=Handshake(io,load_contract('config/plc_contract.yaml'),write_readback=False)
    for tool,address in ((1,40948),(2,40954)):
        io.words.clear(); io.writes.clear(); h.write_reply(1,155,{f'calibration{tool}_xyz':(100,30,-10), 'pc_tool_id':tool})
        assert io.writes == [address, 40935, 40934, 40933]
        assert io.words[40935] == tool and io.words[40934] == 155 and io.words[40933] == 1
        assert all(a in io.words for a in range(address, address+6))
        other = 40954 if tool == 1 else 40948
        assert not any(a in io.words for a in range(40960, 40972))
        assert all(a not in io.words for a in range(other, other+6))


def test_pickacc_status_model(tmp_path):
    from nest_clear.ui.status import MainStatus
    status=MainStatus(tmp_path); status.observe({'event':'pickacc_state','phase':'tool_in_place','tool':2,'message':'工具已到位'})
    assert status.data['pickacc']['phase']=='tool_in_place' and status.data['pickacc']['tool']==2


@pytest.mark.parametrize('scenario', ['table','simple','constant'])
def test_plc_table_simple_and_constant(rig, scenario):
    import hashlib, math
    from nest_clear.tooloffset.commands import send
    from nest_clear.tooloffset.session import nominal_targets
    rig.cfg.pick_accuracy.update(const_max_dev_mm=0.05, const_std_mm=0.025)
    tool=2 if scenario=='simple' else 1
    gradient=lambda x,y:(.002*(x-130),.001*(y-70))
    if tool==2:
        from types import SimpleNamespace
        from nest_clear.tooloffset.store import save_result
        from nest_clear.flow.pick_accuracy import offset_digest
        save_result(rig.root/'calibration'/'tool_offset.yaml',1,{'offset_xy_mm':list(rig.cfg.tools[0][:2])})
        angle=.02; matrix=1.005*np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        nodes=[]
        for i in (2,3,4,5,10,11,12,13,18,19,20,21,26,27,28,29):
            xy=np.array([130.,70.])+matrix@nominal_targets()[i]; dx,dy=gradient(*xy)
            nodes.append(dict(target=f'T{i:02d}',x=float(xy[0]),y=float(xy[1]),dx=float(dx),dy=float(dy),n=1))
        save(rig.root/'calibration'/'pick_compensation.yaml',1,{'type':'table','nodes':nodes,'camera_calibration_sha256':hashlib.sha256(b'test').hexdigest(),'tool_offset_xy':list(rig.cfg.tools[0][:2]),'tool_offset_result_sha256':offset_digest(SimpleNamespace(config=rig.cfg),1)})
    rig.plc.request=(1,150,tool); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    send(rig.root,'pickacc_start'); rig.until(lambda:1 in rig.plc.replies)
    task=1
    for _ in range(100):
        task+=1; rig.plc.request=(task,155,tool); rig.until(lambda:task in rig.plc.replies,timeout=35)
        code,coords=rig.plc.replies[task]
        if code==150: rig.plc.actual=coords['scan_xyz']; continue
        assert code==155, (code,rig.events.status.data['pickacc'])
        assert list(coords)==[f'calibration{tool}_xyz', 'pc_tool_id'] and coords['pc_tool_id']==tool
        e=rig.batch.blackboard.read('pickaccuracy_session')['engine']; target=e.measured[e.current]; xyz=coords[f'calibration{tool}_xyz']
        error=np.array((.08,-.05) if scenario=='constant' else gradient(*target))
        if scenario=='simple': error+=np.array([.02,-.01])
        applied=np.asarray(xyz[:2])-target-np.asarray(rig.cfg.tools[tool-1][:2])
        actual=np.asarray(xyz[:2])+error-applied; rig.plc.actual=(*actual,-10.)
        task+=1; rig.plc.request=(task,156,tool); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='tool_in_place')
        send(rig.root,'pickacc_calculate'); rig.until(lambda:task in rig.plc.replies or rig.events.status.data['pickacc'].get('phase')=='confirm',timeout=35)
        if task not in rig.plc.replies:
            send(rig.root,'pickacc_confirm',{'accept':True}); rig.until(lambda:task in rig.plc.replies)
        code=rig.plc.replies[task][0]
        if code==151: break
        assert code==156, (code,rig.events.status.data['pickacc'])
        task+=1; code,coords=rig.request(task,150,tool); assert code==150; rig.plc.actual=coords['scan_xyz']
    else: pytest.fail('会话未结束')
    result=read(rig.root/'calibration'/'pick_compensation.yaml')[f'tool{tool}']
    assert result['type']==('constant' if scenario=='constant' else 'table')
    if scenario=='simple': assert len(result['samples'])==6


def test_arrival_outside_and_motion_unstable(rig):
    from nest_clear.tooloffset.commands import send
    rig.plc.request=(1,150,1); rig.until(lambda:rig.events.status.data['pickacc'].get('phase')=='select')
    send(rig.root,'pickacc_start'); rig.until(lambda:1 in rig.plc.replies)
    rig.plc.actual=(140,70,-5)
    assert rig.request(2,155)[0]==999
    assert rig.batch.blackboard.read('pickaccuracy_session') is None


def test_outlier_retest_policy(tmp_path):
    def disturbance(n,e):
        if e.current==12 and e.plan[0]=='A': return np.array([.15,0])
        return np.zeros(2)
    e=engine(tmp_path)
    assert run(e,lambda x,y:(.08,-.05),disturbance)=='confirm'
    original=next(s for s in e.samples if s.get('outlier'))
    assert original['discarded'] and original['resolved']
    assert any(s['stage']=='outlier_retest' for s in e.samples)
    assert e.candidate['constant_xy']==pytest.approx([.08,-.05])


def test_fixed_compensation_with_human_noise(tmp_path):
    rng=np.random.default_rng(42); e=engine(tmp_path)
    assert run(e,lambda x,y:(.08,-.05),lambda n,e:rng.normal(0,.01,2))=='confirm'
    assert e.candidate['type']=='constant'
    assert np.linalg.norm(np.asarray(e.candidate['constant_xy'])-[.08,-.05])<.02


def test_repeat_uses_accepted_retest_not_discarded_sample(tmp_path):
    e=engine(tmp_path)
    def disturbance(n,e):
        return np.array([.18,0]) if e.current==12 and e.plan[0]=='A' else np.zeros(2)
    assert run(e,lambda x,y:(.002*(x-130),.001*(y-70)),disturbance)=='confirm', e.failure
    assert e.candidate['type']=='table'


def test_first_outlier_is_retested(tmp_path):
    e=engine(tmp_path)
    assert run(e,lambda x,y:(.08,-.05),lambda n,e:np.array([.16,0]) if n==1 else np.zeros(2))=='confirm'
    assert e.samples[0]['outlier'] and e.samples[0]['discarded']
    assert e.candidate['constant_xy']==pytest.approx([.08,-.05])


def test_constant_verification_falls_back_to_table(tmp_path):
    e=engine(tmp_path)
    def noise(n,e):
        return np.array([.06,0]) if e.plan[0]=='verify' and not getattr(e,'in_b',False) else np.zeros(2)
    assert run(e,lambda x,y:(.08,-.05),noise)=='confirm', e.failure
    assert e.candidate['type']=='table'
    assert sum(bool(s.get('superseded')) for s in e.samples)==2


def field_engine(tool=1, **cfg):
    e=engine(None, tool=tool, min_points_a=5, min_points_b=6, min_infield_corners=0, **cfg)
    e.offsets=((29.330,24.322,0),(30.092,-25.283,0))
    e.predicted={i:np.array([49.1+30*(i//8),-14.2+30*(i%8)]) for i in range(32)}
    e._pose_for=lambda xy, uv: (xy[0]+.86,xy[1]+.92,-5.)
    e._project=lambda xy, pose: np.array([2000.+20*(xy[0]+.86-pose[0]),1500.+20*(xy[1]+.92-pose[1])])
    return e


@pytest.mark.parametrize('tool', [1,2])
def test_field_reachable_constant(tool):
    e=field_engine(tool)
    if tool == 1:
        assert sum(e._within(e._pose_for(e.predicted[i], (2000,1500))) and e._within((*tuple(e.predicted[i]+np.asarray(e.offsets[0][:2])),-10)) for i in (2,3,5,10,12,21,26,28,29)) == 2
    result=e._move_next()
    while result=='continue':
        assert e._within(e.target_pose)
        point=e.predicted[e.current]; e.measured[e.current]=point; e.phase='tool'
        comp=np.asarray(e._verification_comp(e.current,point) if e.plan[0]=='verify' else (0,0))
        xyz=(*tuple(point+np.asarray(e.offsets[tool-1][:2])+comp),-10.)
        assert e._within(xyz)
        e.sent=xyz
        result=e.add_sample((*tuple(np.asarray(xyz[:2])+[.08,-.05]-comp),-10.))
    assert result=='confirm', e.failure
    assert len(e.plans['A'])>=5 and e.candidate['type']=='constant'
    assert 'infield' in e.plans and len(e.plans['verify'][0])==2


def test_field_table_and_failure():
    e=field_engine()
    assert run(e,lambda x,y:(.002*(x-130),.001*(y-70)))=='confirm', e.failure
    assert set(e.plans['B']) == {f'T{i:02d}' for i in e.available}
    assert len(e.plans['repeat'])==3 and e.candidate['type']=='table'
    e=field_engine(); e.travel=((80,100),(-10,10),(-15,0))
    assert run(e,lambda x,y:(0,0))=='failed'
    assert '阶段 A 可用目标不足' in e.failure and '相机 ' in e.failure and '工具 ' in e.failure


def test_table_rejects_collinear_nodes():
    e=field_engine()
    e.b_nodes=[16,17,18,19,20,21]
    e.samples=[{'target':f'T{i:02d}', 'stage':'B', 'target_xy_mm':e.predicted[i].tolist(),
                'residual_xy_mm':[0.,0.]} for i in e.b_nodes]
    with pytest.raises(ValueError, match='共线'):
        e._table()


@pytest.mark.parametrize("code", [90, 95, 100, 110, 130, 150, 155, 156])
def test_invalid_plc_tool_id_gets_999_with_raw_echo(rig, code):
    rows = []
    rig.events.listener = rows.append
    assert rig.request(1, code, 65535) == (999, {"pc_tool_id": 65535})
    assert any(row.get("event") == "error" and "40905=65535 非法" in row.get("reason", "") for row in rows)


def test_request_tool_is_snapshotted_and_session_tool_is_locked(rig):
    from nest_clear.tooloffset.commands import send

    rig.plc.request = (1, 150, 1)
    rig.until(lambda: rig.events.status.data["pickacc"].get("phase") == "select")
    assert rig.events.status.data["pickacc"]["tool"] == 1
    rig.plc.request = (1, 150, 2)
    # A stale UI's old parameter cannot override the PLC's request snapshot.
    send(rig.root, "pickacc_start", {"tool": 2})
    rig.until(lambda: 1 in rig.plc.replies)
    assert rig.plc.replies[1] == (150, {"scan_xyz": (130., 70., -5.), "pc_tool_id": 1})
    assert rig.request(2, 155, 2) == (999, {"pc_tool_id": 2})
    state = rig.events.status.data["pickacc"]
    assert state["tool"] == 1 and state["plc_tool_id"] == 2
    assert "40905 从 1 变为 2" in state["message"]
    assert rig.request(3, 155, 1)[1]["pc_tool_id"] == 1

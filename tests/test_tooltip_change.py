"""无 TCP 的换针对刀握手：模拟相机、命令文件和 PLC 坐标。"""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from autoweaver.motion_policy.nodes.node import Status
from nest_clear.config import load_config
from nest_clear.devices.calib_cameras import CalibCamera
from nest_clear.flow.calibration import ConfirmCalibration, StartCalibration, SaveCompensation, CalibrationFailed
from nest_clear.flow.modes import Mode
from nest_clear.tooltip import ToolTipBaseline, ViewCalib
from nest_clear.tooltip.store import save_baseline, load_baseline, initialize_tools, load_tools
from nest_clear.tooloffset.commands import send

ROOT = Path(__file__).resolve().parents[1]
P0 = (10., 20., 30.)
J = {'front': ((20, 0, 0), (0, 0, 20)), 'side': ((0, 20, 0), (0, 0, 20))}

class Board:
    def __init__(self, **values): self.data = values
    def read(self, key, default=None): return self.data.get(key, default)
    def write(self, key, value, name): self.data[key] = value

class Events:
    def __init__(self, root): self.directory = root; self.run_directory = root / 'run'; self.rows = []
    def emit(self, name, **fields): self.rows.append((name, fields))
    def clock(self): return datetime.now().astimezone()
    def meta_update(self, **kw): pass

@pytest.fixture
def rig(tmp_path):
    config = load_config(ROOT / 'config/nest_clear.yaml')
    config = replace(config, tool_calibration=replace(config.tool_calibration, directory=str(tmp_path)), tools=((1.,2.,3.),(4.,5.,6.)))
    initialize_tools(tmp_path / 'tools.yaml')
    events = Events(tmp_path)
    ctx = SimpleNamespace(config=config, events=events)
    return ctx, tmp_path

def setup(rig, tool=1, pending=False, offset=(0.,0.,0.)):
    ctx, root = rig
    views = [ViewCalib(n, (320,242), None if pending else j) for n,j in J.items()]
    baseline = ToolTipBaseline(f'tool{tool}', P0, views, ctx.config.tools[tool-1], jacobian_pending=pending)
    save_baseline(baseline, root / f'tool{tool}_baseline.yaml', tool=tool, source='test')
    bb = Board(request_func=99+tool, task_id=1, mode=Mode.IDLE, actual=P0)
    start = StartCalibration(ctx); start._blackboard = bb; start.perform()
    bb.write('mode', Mode.CALIBRATING_1 if tool == 1 else Mode.CALIBRATING_2, 'test')
    cameras = {}
    for name, jac in J.items():
        c = CalibCamera(name, {'backend':'mock','target_uv':[320,240], 'p0':P0, 'jacobian':jac, 'tip_offset':offset}, lambda: bb.read('actual'))
        c.open(); cameras[name] = c
    return bb, cameras

def frame(ctx, bb, cameras):
    bb.write('tooltip_images', {n:c.capture() for n,c in cameras.items()}, 'test')
    bb.read('tooltip_session')['images'].append({'raw':{}, 'overlay':{}, 'iteration':len(bb.read('tooltip_session')['images'])+1})
    node = ConfirmCalibration(ctx); node._blackboard = bb
    assert node.on_start() == Status.RUNNING, bb.read('error')
    return node

def command(ctx, node, name):
    send(ctx.events.directory, name)
    return node.on_running()

@pytest.mark.parametrize('tool', [1,2])
def test_pending_probe_confirm_align_and_save(rig, tool):
    from nest_clear.flow.calibration import CheckPosition
    ctx, root = rig; bb, cameras = setup(rig, tool, pending=True, offset=(-.2,-.2,0))
    codes = []
    for index in range(6):
        check = CheckPosition(ctx, False); check._blackboard = bb; check.perform()
        node = frame(ctx, bb, cameras)
        for _ in range(100):
            assert node.on_running() == Status.RUNNING
        assert bb.read('reply_func') is None
        assert command(ctx, node, 'tooltip_confirm_move') == Status.SUCCESS
        codes.append(bb.read('reply_func'))
        target = bb.read('reply_coordinates')[f'calibration{tool}_xyz']
        if index == 4:
            target = (target[0] - .04, target[1], target[2])
        bb.write('actual', target, 'test'); bb.write('reply_func', None, 'test')
    assert codes == [99+tool] + [110 if tool == 1 else 120]*5
    saved = load_baseline(root / f'tool{tool}_baseline.yaml')
    assert not saved.jacobian_pending and all(v.jacobian is not None for v in saved.views)
    assert saved.jacobian_measured_at and list((root/'history').glob('*.yaml'))
    node = frame(ctx, bb, cameras)
    assert node.aligned
    assert command(ctx, node, 'tooltip_confirm_result') == Status.SUCCESS
    assert bb.read('reply_func') == (111 if tool == 1 else 121)
    bb.write('tooltip_final', bb.read('actual'), 'test')
    finish = SaveCompensation(ctx); finish._blackboard = bb; finish.perform()
    assert bb.read('reply_func') == 130
    expected = np.asarray(ctx.config.tools[tool-1])
    assert load_tools(root/'tools.yaml')['tools'][f'tool{tool}']['offset_xyz'] == pytest.approx(expected)
    assert expected == pytest.approx(np.asarray((1,2,3) if tool == 1 else (4,5,6)) + np.asarray(bb.read('actual')) - P0)
    assert np.max(np.abs(expected - (np.asarray((1,2,3) if tool == 1 else (4,5,6)) + [.2,.2,0]))) < .1

def test_existing_jacobian_no_probe_and_limit_abort(rig):
    ctx,_ = rig; bb,cameras = setup(rig, offset=(-1.5,0,0))
    node = frame(ctx,bb,cameras)
    assert node.too_large and not bb.read('tooltip_session')['probing']
    assert command(ctx,node,'tooltip_confirm_move') == Status.RUNNING
    assert bb.read('reply_func') is None
    assert command(ctx,node,'tooltip_abort') == Status.FAILURE
    assert '用户放弃' in bb.read('error')
    failed = CalibrationFailed(ctx); failed._blackboard = bb; failed.perform()
    assert bb.read('reply_func') == 131

class ClickProcess:
    def join(self, timeout=None): pass
    def is_alive(self): return False
class ClickPipe:
    def __init__(self, points):
        from nest_clear.tooltip.click import ClickResult
        self.points = {k: ClickResult(v, 'manual', 1., 0.) for k,v in points.items()}
    def poll(self): return True
    def recv(self): return self.points
    def close(self): pass

def test_click_recalculates_and_changes_decision(rig):
    ctx,_ = rig; bb,cameras = setup(rig, offset=(-1.5,0,0))
    node = frame(ctx,bb,cameras)
    assert node.too_large
    node.process, node.connection = ClickProcess(), ClickPipe({'front':(320,242), 'side':(320,242)})
    assert node.on_running() == Status.RUNNING
    assert node.aligned and not node.too_large
    assert command(ctx,node,'tooltip_confirm_result') == Status.SUCCESS
    assert bb.read('tooltip_session')['rounds'][0]['click'] is True
    assert len(bb.read('tooltip_session')['tip_clicks']) == 2

def test_probe_bad_quality_and_detection_failure(rig):
    ctx,_ = rig; bb,cameras = setup(rig, pending=True)
    for index in range(4):
        node = frame(ctx,bb,cameras)
        assert command(ctx,node,'tooltip_confirm_move') == Status.SUCCESS
        bb.write('actual',bb.read('reply_coordinates')['calibration1_xyz'],'test')
    # Return to P0 with a large image disagreement; centred regression detects the drift.
    cameras['front'].config['tip_offset'] = (3,0,0)
    bb.write('tooltip_images', {n:c.capture() for n,c in cameras.items()}, 'test')
    bb.read('tooltip_session')['images'].append({'raw':{},'overlay':{},'iteration':5})
    node = ConfirmCalibration(ctx); node._blackboard=bb
    assert node.on_start() == Status.FAILURE
    assert '雅可比测量质量不合格' in bb.read('error')
    assert load_baseline(Path(ctx.config.tool_calibration.directory)/'tool1_baseline.yaml').jacobian_pending

def test_probe_detection_failure(rig):
    ctx,_ = rig; bb,cameras = setup(rig, pending=True)
    cameras['front'].config['undetectable'] = True
    bb.write('tooltip_images', {n:c.capture() for n,c in cameras.items()}, 'test')
    bb.read('tooltip_session')['images'].append({'raw':{},'overlay':{},'iteration':1})
    node = ConfirmCalibration(ctx); node._blackboard=bb
    assert node.on_start() == Status.FAILURE
    assert '无法识别针尖' in bb.read('error')

def test_two_corrections_and_position_validation(rig):
    from nest_clear.flow.calibration import CheckPosition
    ctx,_ = rig; bb,cameras = setup(rig, offset=(-.4,-.2,0))
    first = frame(ctx,bb,cameras)
    assert command(ctx,first,'tooltip_confirm_move') == Status.SUCCESS
    target = bb.read('reply_coordinates')['calibration1_xyz']
    # PLC 到位公差内的误差促成第二轮修正。
    bb.write('actual', (target[0]-.04,target[1],target[2]), 'test')
    check = CheckPosition(ctx, False); check._blackboard = bb; check.perform()
    second = frame(ctx,bb,cameras)
    assert not second.aligned
    assert command(ctx,second,'tooltip_confirm_move') == Status.SUCCESS
    assert bb.read('reply_func') == 110
    bb.write('actual', bb.read('reply_coordinates')['calibration1_xyz'], 'test')
    check.perform()
    third = frame(ctx,bb,cameras)
    assert third.aligned
    assert command(ctx,third,'tooltip_confirm_result') == Status.SUCCESS
    assert bb.read('reply_func') == 111
    assert bb.read('tooltip_session')['iterations'] == 2

def test_manual_click_over_limit_recalculates_to_move(rig):
    ctx,_ = rig; bb,cameras = setup(rig, offset=(-1.5,0,0))
    node = frame(ctx,bb,cameras)
    node.process, node.connection = ClickProcess(), ClickPipe({'front':(314,242), 'side':(320,242)})
    assert node.on_running() == Status.RUNNING
    assert not node.too_large and not node.aligned
    assert command(ctx,node,'tooltip_confirm_move') == Status.SUCCESS
    assert bb.read('tooltip_session')['rounds'][0]['click']

def test_address_contract_unchanged():
    from nest_clear.comm import load_contract
    contract = load_contract(ROOT/'config/plc_contract.yaml')
    assert [contract[f'calibration1_{axis}'].address for axis in 'xyz'] == [40948,40950,40952]
    assert [contract[f'calibration2_{axis}'].address for axis in 'xyz'] == [40954,40956,40958]

def test_interrupted_progress_and_result_are_recorded(rig):
    import json
    ctx, root = rig; bb,cameras = setup(rig, offset=(-.2,0,0))
    node = frame(ctx,bb,cameras)
    path = root/'run'/'tooltip'/'1'/'session_record.json'
    assert json.loads(path.read_text())['pending']['stage'] == '对准'
    assert command(ctx,node,'tooltip_confirm_move') == Status.SUCCESS
    record = json.loads(path.read_text())
    assert record['rounds'][0]['confirmed_at'] and record['rounds'][0]['images']

def test_final_coordinate_mismatch_rejected(rig):
    from nest_clear.guard import GuardError
    ctx, root = rig; bb,cameras = setup(rig, offset=(-.2,0,0))
    node = frame(ctx,bb,cameras); command(ctx,node,'tooltip_confirm_move')
    bb.write('actual',bb.read('reply_coordinates')['calibration1_xyz'],'test')
    node = frame(ctx,bb,cameras); command(ctx,node,'tooltip_confirm_result')
    bb.write('tooltip_final',(15.,20.,30.),'test')
    before = (root/'tools.yaml').read_bytes()
    finish=SaveCompensation(ctx); finish._blackboard=bb
    with pytest.raises(GuardError, match='最终坐标不一致'): finish.perform()
    assert (root/'tools.yaml').read_bytes() == before

def test_manual_click_still_over_limit_can_only_abort(rig):
    ctx,_ = rig; bb,cameras = setup(rig, offset=(-1.5,0,0))
    node = frame(ctx,bb,cameras)
    node.process, node.connection = ClickProcess(), ClickPipe({'front':(290,242), 'side':(320,242)})
    assert node.on_running() == Status.RUNNING and node.too_large
    assert command(ctx,node,'tooltip_confirm_move') == Status.RUNNING
    assert bb.read('reply_func') is None
    assert command(ctx,node,'tooltip_abort') == Status.FAILURE
    failed = CalibrationFailed(ctx); failed._blackboard = bb; failed.perform()
    assert bb.read('reply_func') == 131

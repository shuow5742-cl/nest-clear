"""界面对刀基准会话，不使用网络或真实相机。"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import json
import time

import cv2
import numpy as np
import pytest
import yaml

from nest_clear.baseline.session import BaselineSession, DEFAULTS
from nest_clear.tooltip.click import ClickResult
from nest_clear.tooltip.store import initialize_tools, save_tools, load_baseline


class Events:
    def __init__(self, root):
        self.directory = root; self.run_directory = root / 'run'; self.rows = []
    def clock(self): return datetime.now().astimezone()
    def emit(self, event, **kw): self.rows.append((event, kw))


class Cameras:
    def __init__(self): self.opened = False; self.closes = 0; self.x = 320; self.blur = False; self.blank = False
    def open(self): self.opened = True
    def on_stop(self): self.opened = False; self.closes += 1
    def execute(self, payload):
        images = {}
        for name in ('front', 'side'):
            image = np.full((480,640), 255, np.uint8)
            if not self.blank:
                cv2.line(image, (self.x if name == 'front' else 320, 130), (self.x if name == 'front' else 320, 237), 0, 5)
            if name == 'side' and self.blur: image = cv2.GaussianBlur(image, (15,15), 5)
            images[name] = image
        return images


class Plc:
    def __init__(self): self.i = 0; self.unstable = False
    def execute(self, payload):
        assert payload == {'operation':'actual'}
        self.i += 1
        return [1.0 + (self.i % 2 if self.unstable else 0), 2.0, 3.0]


def wait(predicate, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('等待会话超时')


@pytest.fixture
def rig(tmp_path):
    cal = tmp_path / 'cal'; cal.mkdir(); initialize_tools(cal/'tools.yaml'); save_tools(cal/'tools.yaml',1,[1,2,3],source='test')
    calibration = cal/'camera.yaml'; calibration.write_text('camera: test\n')
    (cal/'tool_offset.yaml').write_text(yaml.safe_dump({'tool1':{'offset_xy_mm':[1,2], 'baseline_stale':True},'tool2':{'offset_xy_mm':[4,5], 'baseline_stale':True}}))
    detection = {v: {'direction':[0,1], 'threshold':127, 'min_area':5} for v in ('front','side')}
    cfg = SimpleNamespace(tool_calibration=SimpleNamespace(directory=str(cal), detection=detection, click_timeout_s=2),
                          tool_offset={'settle_samples':3,'settle_window_s':.03,'jitter_tol_mm':.02}, tools=((1,2,3),(0,0,0)))
    cameras, plc, events = Cameras(), Plc(), Events(tmp_path)
    points = iter(())
    def click(image, point, title): return ClickResult(tuple(point), 'manual', 8., 1.)
    session = BaselineSession(cfg,cameras,plc,events,calibration,DEFAULTS,click=click)
    session.submit('baseline_select',{'tool':1},1); wait(lambda: session.phase=='live')
    yield session,cameras,plc,cal,events
    session.end(); session.thread.join(5); assert session.closed


def cmd(s, command, phase, params=None):
    s.submit(command,params or {},len(s.commands)+1); wait(lambda:s.phase==phase)


def test_save_backup_stale_and_other_tool(rig):
    s, camera, plc, cal, events = rig
    old = {'version':1,'tool_name':'tool1','p0':[0,0,0],'offset_xyz':[0,0,0], 'views':[{'name':'front','target_uv':[1,2],'jacobian':None}]}
    (cal/'tool1_baseline.yaml').write_text(yaml.safe_dump(old))
    (cal/'tool2_baseline.yaml').write_text('untouched')
    cmd(s,'baseline_measure','success'); assert s.candidate['p0'] == [1,2,3]
    cmd(s,'baseline_save','confirm'); cmd(s,'baseline_confirm','saved',{'accept':True})
    b=load_baseline(cal/'tool1_baseline.yaml')
    assert b.offset_xyz==(1,2,3) and b.jacobian_pending and all(v.jacobian is None for v in b.views)
    assert list((cal/'history').glob('tool1_baseline*'))
    offset=yaml.safe_load((cal/'tool_offset.yaml').read_text()); assert offset['tool1']['baseline_stale'] is False and offset['tool2']['baseline_stale'] is True
    assert (cal/'tool2_baseline.yaml').read_text()=='untouched'
    s.end(); s.thread.join(5); assert camera.closes == 1
    record=json.loads(next((events.run_directory/'baseline').glob('*/session_record.json')).read_text()); assert record['shots']==1


def test_outside_retry_and_discard(rig):
    s,camera,*_=rig
    camera.x=325; cmd(s,'baseline_measure','fail'); assert '基准标定失败' in s.message and '右侧' in s.message and '画面方向' in s.message
    front_img = cv2.imread(str(s.events.directory / 'live' / 'baseline_front.jpg'))
    assert front_img is not None and front_img.shape[1] > 640
    camera.x=320; cmd(s,'baseline_measure','success'); cmd(s,'baseline_save','confirm'); cmd(s,'baseline_confirm','live',{'accept':False})
    cmd(s,'baseline_measure','success')


def test_manual_and_aux_reset(rig):
    s,*_=rig
    s.click=lambda image, point, title: ClickResult((319.5,239.5) if title=='front' else (321.,240.),'manual',8,1)
    cmd(s,'baseline_manual','success'); assert set(s.candidate['click'])=={'front','side'}
    original=s.candidate['pixels']['front'][:]; p0=s.candidate['p0'][:]
    s.click=lambda image,point,title: ClickResult((315.,235.),'manual',8,1)
    s.submit('baseline_aux',{},10); wait(lambda:s.candidate['pixels']['side']==[315.,235.])
    assert s.candidate['pixels']['front']==original and s.candidate['p0']==p0
    cmd(s,'baseline_save','confirm'); cmd(s,'baseline_confirm','saved',{'accept':True})
    raw=yaml.safe_load((Path(s.config.tool_calibration.directory)/'tool1_baseline.yaml').read_text())
    assert all(view['click'] is True for view in raw['views'])


def test_focus_and_unstable(rig):
    s,camera,plc,*_=rig
    camera.blur=True; cmd(s,'baseline_measure','fail'); assert '模糊' in s.message
    camera.blur=False; plc.unstable=True
    s.submit('baseline_measure',{},9); wait(lambda:'不稳定' in s.message)


def test_camera_exception_releases_devices(tmp_path):
    class Broken(Cameras):
        def execute(self, payload): raise RuntimeError('采图故障')
    cal=tmp_path/'cal'; cal.mkdir(); initialize_tools(cal/'tools.yaml')
    camera_file=cal/'camera.yaml'; camera_file.write_text('x: 1')
    cfg=SimpleNamespace(tool_calibration=SimpleNamespace(directory=str(cal), detection={}), tool_offset={}, tools=((0,0,0),(0,0,0)))
    camera=Broken(); events=Events(tmp_path)
    s=BaselineSession(cfg,camera,Plc(),events,camera_file,DEFAULTS)
    s.submit('baseline_select',{'tool':1},1); s.thread.join(5)
    assert s.closed and camera.closes==1 and '采图故障' in s.message


def test_mode_rejects_every_request_without_ending_session():
    from nest_clear.flow.dispatch import RejectMode, BRANCHES
    from nest_clear.flow.leaves import PrepareError
    from nest_clear.flow.modes import Mode
    from test_hair_integration import Board
    class Log:
        tray_id=None
        def emit(self, *args, **kwargs): pass
    events=Log(); ctx=SimpleNamespace(events=events)
    for func in list(BRANCHES) + [9999]:
        bb=Board(mode=Mode.BASELINE,request_func=func, task_id=1, next_index=0,
                 work_cycle=None,error_diagnostic=None,error_type=None,raw_path=None)
        node=RejectMode(ctx); node._blackboard=bb; node.perform()
        error=PrepareError(events); error._blackboard=bb; error.perform()
        assert bb.read('reply_func') == 999 and bb.read('reply_coordinates') == {}
        assert '对刀基准建立中，请先在界面结束' in bb.read('error')
        assert bb.read('mode') == Mode.BASELINE


def test_wait_request_ignores_start_when_busy(monkeypatch, tmp_path):
    from autoweaver.motion_policy.nodes.node import Status
    from nest_clear.flow.leaves import WaitRequest, CallWorker
    from nest_clear.flow.modes import Mode
    from test_hair_integration import Board
    class Reader:
        last_seq=1
        def poll(self, allowed): return ('baseline_start',{})
    monkeypatch.setattr('nest_clear.flow.tool_offset.reader', lambda ctx: Reader())
    monkeypatch.setattr(CallWorker,'on_running',lambda self: Status.RUNNING)
    events=Events(tmp_path); starts=[]; events.start_baseline=lambda: starts.append(True)
    leaf=WaitRequest(None,SimpleNamespace(),events); leaf._blackboard=Board(mode=Mode.PICKING)
    assert leaf.on_running() == Status.RUNNING
    assert not starts and leaf._blackboard.read('mode')==Mode.PICKING
    assert any('忽略' in data.get('message','') for event,data in events.rows if event=='baseline_state')


def test_manual_rescues_blank_or_blur_and_aux_cancel_preserves_success(rig):
    s, camera, *_ = rig
    camera.blank = True
    cmd(s, 'baseline_measure', 'fail')
    assert '无法识别针尖' in s.message
    s.click = lambda image, point, title: ClickResult((319.5, 239.5) if title == 'front' else (321.0, 240.0), 'manual', 8, 1)
    cmd(s, 'baseline_manual', 'success')
    assert s.candidate['pixels'] == {'front': [319.5, 239.5], 'side': [321.0, 240.0]}
    assert set(s.candidate['click']) == {'front', 'side'}
    s.click = lambda image, point, title: ClickResult(None, 'cancel', 8, 1)
    s.submit('baseline_aux', {}, 20)
    wait(lambda: '已取消' in s.message)
    assert s.phase == 'success'
    cmd(s, 'baseline_save', 'confirm')
    cmd(s, 'baseline_confirm', 'saved', {'accept': True})
    loaded = load_baseline(Path(s.config.tool_calibration.directory) / 'tool1_baseline.yaml')
    assert [list(v.target_uv) for v in loaded.views] == [[319.5, 239.5], [321.0, 240.0]]
    assert all(v.click is True for v in loaded.views)


def test_wait_request_returns_idle_when_baseline_closed(monkeypatch, tmp_path):
    from autoweaver.motion_policy.nodes.node import Status
    from nest_clear.flow.leaves import WaitRequest, CallWorker
    from nest_clear.flow.modes import Mode
    from test_hair_integration import Board

    class Reader:
        last_seq = 0
        def poll(self, allowed): return None

    monkeypatch.setattr('nest_clear.flow.tool_offset.reader', lambda ctx: Reader())
    monkeypatch.setattr(CallWorker, 'on_running', lambda self: Status.RUNNING)
    events = Events(tmp_path)
    events.baseline_session = SimpleNamespace(closed=True)
    leaf = WaitRequest(None, SimpleNamespace(), events)
    leaf._blackboard = Board(mode=Mode.BASELINE)
    assert leaf.on_running() == Status.RUNNING
    assert leaf._blackboard.read('mode') == Mode.IDLE
    assert events.baseline_session is None


def test_command_reader_rejects_baseline_start_during_calibration(tmp_path):
    from nest_clear.tooloffset.commands import CommandReader, send

    events = Events(tmp_path)
    reader = CommandReader(tmp_path, events)
    send(tmp_path, 'baseline_start')
    assert reader.poll({'abort', 'tool_in_place'}) is None
    assert any('忽略' in data.get('message', '') for event, data in events.rows if event == 'baseline_state')


def test_existing_baselines_notice_and_commtest_paths(tmp_path):
    from nest_clear.baseline.session import existing_baselines

    cal = tmp_path / 'commtest_calibration'
    cal.mkdir()
    initialize_tools(cal / 'tools.yaml')
    save_tools(cal / 'tools.yaml', 1, [1.5, -0.5, 2.0], source='test')
    cam_yaml = cal / 'camera.yaml'
    cam_yaml.write_text('camera: test\n')
    (cal / 'tool_offset.yaml').write_text(
        yaml.safe_dump({'tool1': {'offset_xy_mm': [1.5, -0.5], 'time': '2026-11-21T02:00:00+08:00', 'baseline_stale': False}})
    )
    (cal / 'tool1_baseline.yaml').write_text(
        yaml.safe_dump({
            'version': 1, 'tool_name': 'tool1', 'p0': [1, 2, 3], 'offset_xyz': [1.5, -0.5, 2.0],
            'timestamp': '2026-11-21T01:00:00+08:00', 'jacobian_pending': True,
            'views': [{'name': 'front', 'target_uv': [320, 240], 'jacobian': None},
                      {'name': 'side', 'target_uv': [320, 240], 'jacobian': None}],
        })
    )
    cfg = SimpleNamespace(tool_calibration=SimpleNamespace(directory=str(cal)))
    info = existing_baselines(cfg, cam_yaml)
    assert info['tool1']['offset_newer'] is True
    assert info['tool1']['tool_offset_missing'] is False
    assert info['tool1']['offset_xyz'] == [1.5, -0.5, 2.0]
    assert info['tool2']['tool_offset_missing'] is True
    assert info['tool2']['missing'] is True
    fake = object.__new__(BaselineSession)
    assert '工具偏移标定结果比现有对刀基准新' in fake._selection_notice(1, info)
    assert '尚无工具偏移标定结果' in fake._selection_notice(2, info)

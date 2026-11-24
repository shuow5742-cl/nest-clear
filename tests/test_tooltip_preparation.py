from types import SimpleNamespace
from autoweaver.motion_policy.nodes.node import Status
from nest_clear.flow.modes import Mode
from nest_clear.flow.leaves import WaitRequest, CallWorker
from nest_clear.baseline.preparation import Preparation
from test_hair_integration import Board

class Events:
    def __init__(self): self.rows = []; self.baseline_session = None
    def emit(self, event, **data): self.rows.append((event, data))

def test_commands_enter_prepare_and_return_idle(monkeypatch):
    events = Events(); events.tooltip_preparation = Preparation(events)
    events.tooltip_preparation.worker = SimpleNamespace(cameras={})
    monkeypatch.setattr('threading.Thread.start', lambda self: None)
    commands = iter([('tooltip_preview', {'camera':'front', 'on':True}), ('tooltip_preview', {'camera':'side', 'on':True}), ('tooltip_preview', {'camera':'front', 'on':False}), ('tooltip_preview', {'camera':'side', 'on':False})])
    reader = SimpleNamespace(last_seq=1, poll=lambda allowed: next(commands))
    monkeypatch.setattr('nest_clear.flow.tool_offset.reader', lambda ctx: reader)
    monkeypatch.setattr(CallWorker, 'on_running', lambda self: Status.RUNNING)
    leaf = WaitRequest(None, SimpleNamespace(), events); leaf._blackboard = Board(mode=Mode.IDLE)
    for expected in (Mode.BASELINE, Mode.BASELINE, Mode.BASELINE, Mode.IDLE):
        assert leaf.on_running() == Status.RUNNING
        assert leaf._blackboard.read('mode') == expected


def test_independent_preview_and_exposure(tmp_path, monkeypatch):
    import time
    import numpy as np
    from nest_clear.devices.tooltip_camera_store import load_settings
    class Camera:
        def __init__(self, name, fail=False):
            self.config = {'device':name, 'settings_path':str(tmp_path/'cameras.yaml')}; self.fail=fail; self.exposure=100
        def open(self):
            if self.fail: raise OSError('USB 带宽不足')
        def close(self): pass
        def capture(self): time.sleep(.01); return np.full((60,80), 210, np.uint8)
        def get_exposure(self): return self.exposure
        def set_exposure(self, value): self.exposure=value
        def settings_snapshot(self): return {'device':self.config['device'], 'controls': {'exposure': self.exposure}}
    events=Events(); events.directory=tmp_path
    worker=SimpleNamespace(cameras={'front':Camera('front'), 'side':Camera('side', True)})
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    measurements=[]
    def measure(camera, **kwargs):
        measurements.append(1)
        return {'exposure':100, 'p99':210, 'saturated_fraction':0, 'contrast':100}
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', measure)
    prep=Preparation(events, worker)
    prep.command('tooltip_preview', {'camera':'front', 'on':True})
    prep.command('tooltip_preview', {'camera':'side', 'on':True})
    deadline=time.monotonic()+3
    while time.monotonic()<deadline and not ((tmp_path/'live/baseline_front.jpg').exists() and prep.state['side'].get('error')): time.sleep(.01)
    assert (tmp_path/'live/baseline_front.jpg').exists()
    assert 'USB 带宽不足' in prep.state['side']['error']
    assert load_settings(tmp_path/'cameras.yaml')['cameras']['front']['measured']
    prep.command('tooltip_exposure_retest', {'camera':'front'})
    deadline=time.monotonic()+3
    while len(measurements)<2 and time.monotonic()<deadline: time.sleep(.01)
    assert len(measurements)==2
    prep.close()
    for thread in prep.threads.values(): thread.join(2)


def test_handoff_keeps_camera_open(tmp_path, monkeypatch):
    import time
    import numpy as np
    class Camera:
        config={'device':'front', 'settings_path':str(tmp_path/'cameras.yaml')}
        def __init__(self): self.opens=0; self.closes=0
        def open(self): self.opens+=1
        def close(self): self.closes+=1
        def capture(self): time.sleep(.01); return np.full((60,80), 210, np.uint8)
        def get_exposure(self): return 100
        def set_exposure(self, n): pass
        def settings_snapshot(self): return {'device':'front'}
    camera=Camera(); events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', lambda c, **kw: {'exposure':100,'p99':210,'saturated_fraction':0,'contrast':100})
    prep=Preparation(events, SimpleNamespace(cameras={'front':camera,'side':Camera()}))
    prep.command('tooltip_preview', {'camera':'front','on':True})
    deadline=time.monotonic()+2
    while not camera.opens and time.monotonic()<deadline: time.sleep(.01)
    prep.handoff(); prep.await_handoff()
    assert camera.opens==1 and camera.closes==0
    camera.close()


def test_focus_improves_and_tip_motion_resets_peak(tmp_path, monkeypatch):
    import time
    import cv2
    import numpy as np
    from nest_clear.devices.tooltip_camera_store import save_measurement
    path=tmp_path/'cameras.yaml'
    save_measurement(path, 'front', {'device':'front'}, {'exposure':100,'p99':210,'saturated_fraction':0,'contrast':100})
    class Camera:
        config={'device':'front', 'settings_path':str(path)}
        def __init__(self): self.sigma=3.; self.x=100
        def open(self): pass
        def close(self): pass
        def get_exposure(self): return 100
        def settings_snapshot(self): return {'device':'front'}
        def capture(self):
            time.sleep(.02)
            image=np.full((240,320), 230, np.uint8)
            cv2.line(image, (self.x,30), (self.x,120), 20, 9)
            return cv2.GaussianBlur(image, (0,0), self.sigma)
    camera=Camera(); events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    monkeypatch.setattr('nest_clear.baseline.preparation.auto_exposure', lambda *a,**kw: (_ for _ in ()).throw(AssertionError('already measured')))
    prep=Preparation(events, SimpleNamespace(cameras={'front':camera,'side':Camera()}))
    prep.command('tooltip_preview', {'camera':'front','on':True})
    def wait(predicate):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if predicate(): return
            time.sleep(.02)
        assert False, prep.state['front']
    wait(lambda: prep.state['front'].get('tenengrad') is not None)
    blurry=prep.state['front']['tenengrad']; camera.sigma=.5
    wait(lambda: prep.state['front'].get('tenengrad',0)>blurry*1.1)
    sharp_peak=prep.state['front']['peak']; camera.sigma=3.; camera.x=190
    wait(lambda: (prep.state['front'].get('tip_uv') or [0])[0]>150)
    assert prep.state['front']['peak'] < sharp_peak
    camera.sigma=.5
    wait(lambda: prep.state['front'].get('peak',0) > blurry*1.1)
    camera.sigma=3.
    wait(lambda: prep.state['front'].get('percent',100) < 95)
    prep.command('tooltip_peak_reset', {'camera':'front'})
    wait(lambda: prep.state['front'].get('percent',0) >= 99)
    assert not any(event=='tooltip_exposure_failed' for event,_ in events.rows)
    prep.close()
    prep.threads['front'].join(2)


def test_slow_open_does_not_block_tick(tmp_path, monkeypatch):
    import threading
    import time
    gate=threading.Event()
    class Slow:
        config={'device':'x', 'settings_path':str(tmp_path/'cameras.yaml')}
        def open(self): gate.wait(2); raise RuntimeError('模拟打开失败')
        def close(self): pass
    events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    prep=Preparation(events, SimpleNamespace(cameras={'front':Slow(),'side':Slow()})); events.tooltip_preparation=prep
    reader=SimpleNamespace(last_seq=1, poll=lambda allowed: ('tooltip_preview',{'camera':'front','on':True}))
    monkeypatch.setattr('nest_clear.flow.tool_offset.reader',lambda ctx:reader)
    monkeypatch.setattr(CallWorker,'on_running',lambda self: Status.RUNNING)
    leaf=WaitRequest(None, SimpleNamespace(), events); leaf._blackboard=Board(mode=Mode.IDLE)
    begin=time.monotonic(); assert leaf.on_running()==Status.RUNNING
    assert time.monotonic()-begin < .2 and leaf._blackboard.read('mode') == Mode.BASELINE
    gate.set(); prep.close(); prep.threads['front'].join(2)


def test_assignment_and_swap_update_store(tmp_path, monkeypatch):
    import time
    from nest_clear.devices.tooltip_camera_store import load_settings
    a='/dev/v4l/by-path/pci-usb-0:4:1.0-video-index0'
    b='/dev/v4l/by-path/pci-usb-0:8:1.0-video-index0'
    path=tmp_path/'cameras.yaml'
    cameras={name:SimpleNamespace(config={'settings_path':str(path),'device':'待接入'}) for name in ('front','side')}
    events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda:[a,b])
    prep=Preparation(events, SimpleNamespace(cameras=cameras))
    prep.devices=[a,b]
    def wait():
        until=time.monotonic()+3
        while prep.busy and time.monotonic()<until: time.sleep(.01)
        assert not prep.busy
    prep.command('tooltip_assign', {'camera':'front','device':a}); wait()
    prep.command('tooltip_assign', {'camera':'side','device':b}); wait()
    prep.command('tooltip_swap', {}); wait()
    assert load_settings(path)['devices']=={'front':b,'side':a}
    assert len(list((tmp_path/'history').glob('*.yaml'))) == 2
    assert len([row for row in events.rows if row[0]=='tooltip_camera_assigned']) == 3


def test_worker_reuses_handed_off_device(tmp_path, monkeypatch):
    import time
    import numpy as np
    from nest_clear.workers.calib_camera import CalibCameraWorker
    from nest_clear.devices.tooltip_camera_store import save_measurement
    path=tmp_path/'settings.yaml'
    save_measurement(path, 'front', {'device':'front'}, {'exposure':100,'p99':220,'saturated_fraction':0,'contrast':100})
    class Camera:
        def __init__(self,name): self.config={'device':name,'settings_path':str(path)}; self.device=None; self.opens=0
        def open(self): self.opens+=1; self.device=object()
        def close(self): self.device=None
        def capture(self): time.sleep(.01); return np.full((60,80),220,np.uint8)
        def get_exposure(self): return 100
        def settings_snapshot(self): return {'device':self.config['device']}
    events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices',lambda:[])
    worker=CalibCameraWorker({'front':{'backend':'mock'},'side':{'backend':'mock'}})
    worker.cameras={'front':Camera('front'),'side':Camera('side')}
    prep=Preparation(events,worker)
    prep.command('tooltip_preview',{'camera':'front','on':True})
    until=time.monotonic()+2
    while worker.cameras['front'].device is None and time.monotonic()<until: time.sleep(.01)
    prep.handoff(); prep.await_handoff(); worker.open()
    assert worker.cameras['front'].opens == 1
    assert worker.cameras['side'].opens == 1
    worker.on_stop()


def test_daily_report_includes_preparation(tmp_path):
    import json
    from datetime import date
    from nest_clear.record.report import report
    day=date(2026,11,23); directory=tmp_path/day.isoformat(); directory.mkdir()
    row={'event':'tooltip_preparation_finished','wall':'2026-11-23T10:00:00','cameras':{'front':{'edge_width_um':12.3},'side':{'edge_width_um':20.0}}}
    (directory/'events.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
    text=report(tmp_path,day)
    assert '对刀相机准备：1 次' in text and '12.3' in text and '20.0' in text


def test_plc_request_rejected_without_stopping_previews():
    from nest_clear.flow.dispatch import RejectMode
    from nest_clear.flow.leaves import PrepareError
    events=Events(); events.tray_id=None
    prep=Preparation(events); prep.enabled['front']=True; prep.enabled['side']=True
    board=Board(mode=Mode.BASELINE, request_func=100, task_id=3, next_index=0,
                work_cycle=None, error_diagnostic=None, error_type=None, raw_path=None)
    node=RejectMode(SimpleNamespace(events=events)); node._blackboard=board; node.perform()
    error=PrepareError(events); error._blackboard=board; error.perform()
    assert board.read('reply_func') == 999 and prep.enabled == {'front':True,'side':True}
    assert '请先在界面结束' in board.read('error')


def test_failed_preview_retries_on_single_command(tmp_path, monkeypatch):
    import time
    import numpy as np
    from nest_clear.devices.tooltip_camera_store import save_measurement
    path = tmp_path/'cameras.yaml'
    save_measurement(path, 'front', {'device':'front'}, {'exposure':100,'p99':210,'saturated_fraction':0,'contrast':100})
    class Camera:
        config={'device':'front', 'settings_path':str(path)}
        def __init__(self): self.opens=0
        def open(self):
            self.opens += 1
            if self.opens == 1: raise OSError('USB 带宽不足')
        def close(self): pass
        def get_exposure(self): return 100
        def settings_snapshot(self): return {'device':'front'}
        def capture(self): time.sleep(.01); return np.full((60,80),210,np.uint8)
    camera=Camera(); events=Events(); events.directory=tmp_path
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: [])
    prep=Preparation(events, SimpleNamespace(cameras={'front':camera,'side':camera}))
    prep.command('tooltip_preview', {'camera':'front','on':True})
    deadline=time.monotonic()+3
    while not prep.state['front'].get('error') and time.monotonic()<deadline: time.sleep(.01)
    assert prep.state['front'].get('error')
    prep.command('tooltip_preview', {'camera':'front','on':True})
    while camera.opens < 2 and time.monotonic()<deadline: time.sleep(.01)
    assert camera.opens == 2 and prep.state['front'].get('error') is None
    prep.close(); prep.threads['front'].join(2)


def test_dual_preview_10s_throttles_events_and_updates_state_file(tmp_path, monkeypatch):
    import json
    import threading
    import time
    from datetime import datetime, timedelta, timezone
    import numpy as np
    from nest_clear.devices.tooltip_camera_store import save_measurement
    from nest_clear.paths import main_status_file, tooltip_preview_file
    from nest_clear.record.events import EventRecorder

    settings_path = tmp_path / 'cameras.yaml'
    for name in ('front', 'side'):
        save_measurement(settings_path, name, {'device': name}, {'exposure': 100, 'p99': 210, 'saturated_fraction': 0, 'contrast': 100})

    clock_lock = threading.Lock()
    virtual_mono = [100.0]
    base_wall = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
    barrier = threading.Barrier(2, timeout=5)
    target_frames = 51  # 51 * 0.2s = 10.2s (> 10s at 5Hz)
    done = threading.Event()
    counts = {'front': 0, 'side': 0}

    def fake_mono():
        with clock_lock:
            return virtual_mono[0]

    def fake_clock():
        with clock_lock:
            return base_wall + timedelta(seconds=virtual_mono[0] - 100.0)

    class Camera:
        def __init__(self, name):
            self.name = name
            self.config = {'device': name, 'settings_path': str(settings_path)}
        def open(self): pass
        def close(self): pass
        def get_exposure(self): return 100
        def settings_snapshot(self): return {'device': self.name}
        def capture(self):
            idx = barrier.wait()
            with clock_lock:
                if idx == 0:
                    virtual_mono[0] += 0.2
                counts[self.name] += 1
                if counts['front'] >= target_frames and counts['side'] >= target_frames:
                    done.set()
            return np.full((60, 80), 210, np.uint8)

    events = EventRecorder(tmp_path, process='main', run=True, clock=fake_clock, monotonic=fake_mono)
    monkeypatch.setattr('nest_clear.baseline.preparation.enumerate_capture_devices', lambda: ['front', 'side'])
    prep = Preparation(events, SimpleNamespace(cameras={'front': Camera('front'), 'side': Camera('side')}),
                       clock=fake_clock, monotonic=fake_mono)

    preview_path = tooltip_preview_file(tmp_path)
    replace_calls = []
    real_replace = __import__('os').replace
    def spy_replace(src, dst):
        if str(dst) == str(preview_path):
            replace_calls.append(fake_mono())
        return real_replace(src, dst)
    monkeypatch.setattr('nest_clear.baseline.preparation.os.replace', spy_replace)

    prep.command('tooltip_preview', {'camera': 'front', 'on': True})
    prep.command('tooltip_preview', {'camera': 'side', 'on': True})
    assert done.wait(10), counts
    # Give threads a moment to finish writing frame 51 before closing.
    time.sleep(0.05)
    live_doc = json.loads(preview_path.read_text(encoding='utf-8'))
    for cam in ('front', 'side'):
        cam_live = live_doc['cameras'][cam]
        assert {'tenengrad', 'peak', 'percent', 'edge_width_um', 'edge_width_px', 'contrast', 'source', 'history', 'fps', 'exposure_ms', 'updated_at'} <= set(cam_live)
        assert len(cam_live['history']) >= 10
    prep.command('tooltip_preview', {'camera': 'front', 'on': False})
    prep.command('tooltip_preview', {'camera': 'side', 'on': False})
    prep.close()
    for t in prep.threads.values():
        t.join(2)
    barrier.abort()
    if events.status and events.status.timer:
        events.status.timer.cancel()
        events.status.flush()

    events_file = events.run_directory / 'events.jsonl'
    rows = [json.loads(line) for line in events_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    summaries = [r for r in rows if r['event'] == 'tooltip_focus_summary']
    baseline_rows = [r for r in rows if r['event'] == 'baseline_state']
    preview_related = [r for r in rows if r['event'].startswith(('baseline_state', 'tooltip_preview_', 'tooltip_exposure_', 'tooltip_camera_', 'tooltip_preparation_', 'tooltip_focus_'))]

    assert 2 <= len(summaries) <= 4
    for cam in ('front', 'side'):
        cam_summaries = [r for r in summaries if r['camera'] == cam]
        assert 1 <= len(cam_summaries) <= 2
    for row in summaries:
        assert {'current', 'peak', 'percent', 'edge_width_um', 'source', 'camera'} <= set(row)
        assert 'history' not in row
    for row in baseline_rows:
        for cam_state in (row.get('preview') or {}).values():
            assert not ({'tenengrad', 'peak', 'percent', 'edge_width_um', 'history', 'fps', 'exposure_ms'} & set(cam_state))
    assert len(baseline_rows) <= 15
    assert len(preview_related) <= 25

    # State file was updated continuously across the 10s run and <= 5Hz
    assert len(replace_calls) >= 20
    main_data = json.loads(main_status_file(tmp_path).read_text(encoding='utf-8'))
    for cam in ('front', 'side'):
        assert 'device' in main_data['baseline']['preview'][cam]
        assert 'percent' not in main_data['baseline']['preview'][cam]

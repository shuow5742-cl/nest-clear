import json
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nest_clear.config import load_config
from nest_clear.paths import live_image_file, main_status_file, supervisor_state_file
from nest_clear.supervisor.runner import Supervisor
from nest_clear.record.live import write_live
from nest_clear.ui.status import MainStatus

class PLC:
    def read(self, name): return {'status': 5, 'algorithm_ready': 1}.get(name, 0)
    def write(self, name, value): pass
    def close(self): pass

def test_supervisor_state(tmp_path):
    cfg=replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    s=Supervisor(cfg,PLC()); s.tick(); s._publish(); state=json.loads(supervisor_state_file(tmp_path).read_text())
    assert state['pid']==os.getpid() and state['word']==5 and state['plc_connected'] is True
    assert state['plc_last_read'] and state['heartbeat_last_write'] and state['algorithm_ready']==1
    assert state['child_state']=='未启动'

def test_status_and_images(tmp_path):
    status=MainStatus(tmp_path)
    for row in ({'event':'mode_change','mode':'挑毛中'},{'event':'request','task_id':3,'func':20},{'event':'reply','reply_func':999,'wall':'now'},{'event':'camcal_step','sequence':1,'plan':[[1,2,3]]},{'event':'tooltip_detect','delta':[1,2,3]}): status.observe(row)
    status.flush(); data=json.loads(main_status_file(tmp_path).read_text()); assert data['mode']=='挑毛中' and data['task_id']==3 and data['last_reply_func']==999
    assert data['camcal']['sequence']==1 and data['tooltip']['history'][0]['delta']==[1,2,3]
    for name in ('camcal_latest','calib_front_latest','calib_side_latest'):
        write_live(tmp_path,name,np.zeros((8,8,3),dtype=np.uint8)); assert live_image_file(tmp_path,name).is_file()

def test_ui_restarts_and_gives_up(tmp_path,monkeypatch):
    cfg=replace(load_config('config/nest_clear.yaml'),record_directory=str(tmp_path)); s=Supervisor(cfg,PLC())
    clock=[100.0]; monkeypatch.setattr('nest_clear.supervisor.runner.time.monotonic',lambda:clock[0]); starts=[]
    class Process:
        pid=99
        def poll(self): return 1
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen',lambda *a,**k: starts.append(a) or Process())
    s._ui_tick()
    for _ in range(5):
        s._ui_tick(); clock[0]+=5; s._ui_tick()
    assert len(starts)==5 and s._ui_disabled

@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='无 DISPLAY')
def test_ui_smoke(tmp_path, monkeypatch):
    import tkinter
    from nest_clear.ui.app import App
    try:
        probe = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip('DISPLAY 不可连接')
    probe.destroy()
    from nest_clear.ui.demo import write_demo
    write_demo(tmp_path, initial=True)
    app = App(tmp_path, 50, page='camcal')
    app.root.geometry('1366x768')
    try:
        for i in range(app.tabs.index('end')):
            app.tabs.select(i); app.refresh(); app.root.update()
        assert app.tabs.index('end') == 6
        class Screenshot:
            def save(self, path): path.write_bytes(b'png')
        monkeypatch.setattr('nest_clear.ui.app.ImageGrab.grab', lambda **kwargs: Screenshot())
        target = tmp_path / 'screen.png'
        app.capture_and_exit(target)
        assert target.read_bytes() == b'png'
    finally:
        try: app.root.destroy()
        except tkinter.TclError: pass

def test_no_display_does_not_start_ui(tmp_path, monkeypatch):
    cfg=replace(load_config('config/nest_clear.yaml'),record_directory=str(tmp_path)); s=Supervisor(cfg,PLC())
    monkeypatch.delenv('DISPLAY',raising=False)
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen',lambda *a,**k: pytest.fail('不应启动界面'))
    monkeypatch.setattr(s,'tick',lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    s.run(); assert s._ui_disabled


def test_ui_system_python_imports():
    import subprocess
    import sys
    from pathlib import Path
    python = Path('/usr/bin/python3')
    if not python.exists(): pytest.skip('/usr/bin/python3 不存在')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    result = subprocess.run([str(python), '-c', "import nest_clear.ui.app, nest_clear.ui.model, nest_clear.ui.demo, nest_clear.ui.status; import sys; assert not any(m in sys.modules for m in ('numpy','cv2','autoweaver'))"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def test_supervisor_ui_python_and_source_path(tmp_path, monkeypatch):
    from pathlib import Path
    cfg=replace(load_config('config/nest_clear.yaml'), record_directory=str(tmp_path))
    cfg=replace(cfg, ui=replace(cfg.ui, python='/custom/python'))
    supervisor=Supervisor(cfg,PLC()); captured={}
    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr('nest_clear.supervisor.runner.subprocess.Popen', start)
    monkeypatch.setenv('PYTHONPATH', '/existing/path')
    supervisor._ui_tick()
    assert captured['command'][:3] == ['/custom/python','-m','nest_clear.ui']
    assert captured['env']['PYTHONPATH'].split(os.pathsep) == [str(Path(__file__).resolve().parents[1] / 'src'), '/existing/path']

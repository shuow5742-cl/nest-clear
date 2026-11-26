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
@pytest.mark.parametrize('geometry', ['1366x768', '1920x1080'])
def test_home_layout_with_system_python(tmp_path, geometry):
    import subprocess
    import tkinter
    from pathlib import Path
    python = Path('/usr/bin/python3')
    if not python.exists():
        pytest.skip('系统 Python 不存在')
    try:
        probe = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip('DISPLAY 不可连接')
    probe.destroy()
    code = '''
import sys, time
from nest_clear.ui.app import App
from nest_clear.ui.demo import write_demo
write_demo(sys.argv[1], initial=True)
app = App(sys.argv[1], 50, page='main', ignore_before_ns=0)
try:
    app.root.state('normal')
    app.root.attributes('-zoomed', False)
    app.root.geometry(sys.argv[2])
    app.refresh()
    for index in range(4):
        app.table.insert('', 'end', iid=f'test-{index}')
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    for content in (app.bits[0].master, app.pick_values['盘号'].master):
        body = content.master
        for widget in content.winfo_children():
            bottom = widget.winfo_rooty() + widget.winfo_height()
            for ancestor in (body, body.master, body.master.master, app.home):
                assert bottom <= ancestor.winfo_rooty() + ancestor.winfo_height(), (widget, ancestor)
    for table, count in ((app.process_table, 2), (app.reply_table, 5), (app.table, 4)):
        bbox = table.bbox(table.get_children()[count - 1])
        assert bbox and bbox[1] + bbox[3] <= table.winfo_height(), (table, bbox)
    if sys.argv[2] == '1920x1080':
        assert app.bits[3].winfo_y() - app.bits[0].winfo_y() == 30
finally:
    app.root.destroy()
'''
    result = subprocess.run([str(python), '-c', code, str(tmp_path), geometry],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='无 DISPLAY')
@pytest.mark.parametrize('geometry', ['1366x768', '1680x1050', '1920x1080'])
def test_home_tables_fit_window(tmp_path, geometry):
    import tkinter
    from nest_clear.ui.app import App
    from nest_clear.ui.demo import write_demo
    try:
        probe = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip('DISPLAY 不可连接')
    probe.destroy()
    write_demo(tmp_path, initial=True)
    app = App(tmp_path, 50, page='main', ignore_before_ns=0)
    try:
        try:
            app.root.state('normal')
            app.root.attributes('-zoomed', False)
        except tkinter.TclError:
            pass
        app.root.geometry(geometry)
        app.refresh()
        for index in range(4):
            app.table.insert('', 'end', iid=f'test-{index}')
        for _ in range(5):
            app.root.update()
        for table, count in ((app.process_table, 2), (app.reply_table, 5), (app.table, 4)):
            row = table.get_children()[count - 1]
            bbox = table.bbox(row)
            assert bbox and bbox[1] + bbox[3] <= table.winfo_height()
        for table in (app.process_table, app.reply_table):
            bbox = table.bbox(table.get_children()[-1])
            assert table.winfo_height() - bbox[1] - bbox[3] <= 4
        from tkinter import font as tkfont
        assert tkfont.Font(root=app.root, font=app.bits[0].cget('font')).actual('size') == 12
        assert tkfont.Font(root=app.root, font=app.pick_values['盘号'].cget('font')).actual('size') == 12
        for widgets in (app.bits[::3], list(app.pick_values.values())[::2]):
            assert all(22 <= second.winfo_y() - first.winfo_y() <= 34
                       for first, second in zip(widgets, widgets[1:]))
            content = widgets[0].master
            body = content.master
            assert content.winfo_y() == 0
            assert content.winfo_height() <= body.winfo_height()
            for widget in content.winfo_children():
                bottom = widget.winfo_rooty() + widget.winfo_height()
                for ancestor in (body, body.master, body.master.master, app.home):
                    assert bottom <= ancestor.winfo_rooty() + ancestor.winfo_height()
        if geometry == '1920x1080':
            assert app.table.winfo_height() >= 26 * 7
    finally:
        app.root.destroy()


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
    app = App(tmp_path, 50, page='camcal', ignore_before_ns=0)
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


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='无 DISPLAY')
def test_restart_ignores_old_pictures_until_updated(tmp_path):
    import time
    import tkinter
    from PIL import Image
    from nest_clear.ui.app import App
    from nest_clear.ui.demo import write_demo
    try:
        probe = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip('DISPLAY 不可连接')
    probe.destroy()
    write_demo(tmp_path, initial=True)
    app = App(tmp_path, 50, page='main')
    try:
        app.refresh()
        for _ in range(4):
            app.poll_pictures()
            app.root.update()
            time.sleep(0.02)
        assert all(pic.image is None and pic.cget('text') == '暂无图像' for pic in app.pictures)
        assert len(app.reply_table.get_children()) == 5
        assert app.pick_values['盘号'].cget('text') == '1'
        path = live_image_file(tmp_path, 'latest')
        Image.new('RGB', (32, 24), 'red').save(path, format='JPEG')
        for _ in range(15):
            app.poll_pictures()
            app.root.update()
            if app.pictures[0].image is not None:
                break
            time.sleep(0.02)
        assert app.pictures[0].image is not None
        assert app.pictures[0].cget('text') == ''
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
    result = subprocess.run([str(python), '-c', "import sys; sys.modules.update(dict.fromkeys(('cv2','numpy','yaml'))); import nest_clear.ui.app, nest_clear.ui.model, nest_clear.ui.demo, nest_clear.ui.status; assert 'autoweaver' not in sys.modules"], env=env, capture_output=True, text=True)
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

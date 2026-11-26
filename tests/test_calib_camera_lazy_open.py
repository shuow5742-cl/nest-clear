"""对刀相机断开后可在后续请求中重新打开。"""
import pytest

from nest_clear.workers.calib_camera import CalibCameraWorker


class Events:
    run_directory = None

    def __init__(self):
        self.rows = []

    def emit(self, name, **fields):
        self.rows.append((name, fields))


def test_open_failure_retries_on_next_request(monkeypatch):
    events = Events()
    worker = CalibCameraWorker({n: {"backend": "mock"} for n in ("front", "side")}, events=events)
    front = worker.cameras["front"]
    original = front.open
    def disconnected():
        raise OSError("设备不存在")
    monkeypatch.setattr(front, "open", disconnected)
    with pytest.raises(Exception, match="对刀相机未接入或打不开") as error:
        worker.execute({})
    from nest_clear.guard import diagnostic
    assert "对刀相机未接入或打不开" in diagnostic(str(error.value))["what"]
    assert "请接好正面/侧面相机后重试" in diagnostic(str(error.value))["what"]
    assert not worker.opened
    assert any("设备不存在" in row[1].get("reason", "") for row in events.rows)
    monkeypatch.setattr(front, "open", original)
    assert set(worker.execute({})) == {"front", "side"}
    worker.on_stop()


def test_capture_disconnect_reopens(monkeypatch):
    worker = CalibCameraWorker({n: {"backend": "mock"} for n in ("front", "side")})
    worker.open()
    side = worker.cameras["side"]
    original = side.capture
    monkeypatch.setattr(side, "capture", lambda: (_ for _ in ()).throw(OSError("掉线")))
    with pytest.raises(Exception, match="对刀相机未接入或打不开"):
        worker.execute({})
    assert not worker.opened
    monkeypatch.setattr(side, "capture", original)
    assert set(worker.execute({})) == {"front", "side"}
    worker.on_stop()


def test_plc_request_fails_then_retries(rig, monkeypatch):
    """真实握手：先回 131，恢复连接后下一请求进入对刀流程。"""
    from test_e2e_calibration import P0
    import json
    plc, app, path = rig
    assert plc.io.read("algorithm_ready") == 1
    front = app.calib_camera.cameras["front"]
    original = front.open
    monkeypatch.setattr(front, "open", lambda: (_ for _ in ()).throw(OSError("未连接")))
    plc.set_actual_xyz(P0)
    task = plc.request(100, tool_id=2)
    assert plc.wait_reply(task, 5)["func_code"] == 131
    rows = [json.loads(line) for line in next((path / "runs").glob("*/*/events.jsonl")).read_text().splitlines()]
    assert any("对刀相机" in str(row) and "未连接" in str(row) for row in rows)
    monkeypatch.setattr(front, "open", original)
    assert plc.start_tool_calibration(1, P0, timeout_s=5)[-1]["func_code"] == 111


from test_e2e_calibration import rig


def test_timeout_retries_after_capture_returns(monkeypatch):
    import threading
    worker = CalibCameraWorker({n: {"backend": "mock"} for n in ("front", "side")}, timeout_s=.02)
    camera = worker.cameras["front"]
    original = camera.capture
    release = threading.Event()
    done = threading.Event()
    def slow():
        release.wait(2)
        done.set()
        return original()
    monkeypatch.setattr(camera, "capture", slow)
    with pytest.raises(Exception, match="对刀相机"):
        worker.execute({})
    with pytest.raises(Exception, match="稍后重试"):
        worker.execute({})
    monkeypatch.setattr(camera, "capture", original)
    release.set()
    assert done.wait(1)
    assert worker.capture_done["front"].wait(1)
    assert set(worker.execute({})) == {"front", "side"}
    worker.on_stop()


def test_supervisor_owned_camera_returns_131_and_retries_after_release(rig):
    from nest_clear.devices.calib_cameras import CalibCamera
    from test_e2e_calibration import P0
    import json
    plc, app, path = rig
    front = app.calib_camera.cameras['front']
    owner = CalibCamera('front', dict(front.config), lambda: [0, 0, 0])
    owner.open()
    try:
        plc.set_actual_xyz(P0)
        task = plc.request(100, tool_id=2)
        assert plc.wait_reply(task, 5)['func_code'] == 131
        rows = [json.loads(line) for line in next((path / 'runs').glob('*/*/events.jsonl')).read_text().splitlines()]
        assert any('仍被父程序预览占用' in str(row) for row in rows)
    finally:
        owner.close()
    assert plc.start_tool_calibration(1, P0, timeout_s=5)[-1]['func_code'] == 111


def test_one_auto_camera_causes_plc_reply_131(rig, monkeypatch):
    from test_e2e_calibration import P0
    plc, app, path = rig
    app.calib_camera.on_stop()
    for camera in app.calib_camera.cameras.values():
        camera.backend = 'uvc'
    def missing(*args):
        raise ValueError('只检测到 1 台对刀相机（usb-0:6），需两台都接上才能区分正/侧')
    monkeypatch.setattr('nest_clear.workers.calib_camera.resolve_tooltip_roles', missing)
    plc.set_actual_xyz(P0)
    task = plc.request(100, tool_id=2)
    assert plc.wait_reply(task, 5)['func_code'] == 131
    import json
    rows = [json.loads(line) for line in next((path / 'runs').glob('*/*/events.jsonl')).read_text().splitlines()]
    assert any('只检测到 1 台对刀相机' in str(row) for row in rows)

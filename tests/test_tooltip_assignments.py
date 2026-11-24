from nest_clear.devices.tooltip_camera_store import load_settings, save_assignments, save_measurement, measurement_for_device
from nest_clear.ui.model import classify_event
from nest_clear.ui.status import MainStatus


def test_assignments_backup_and_measurement_follows_device(tmp_path):
    path = tmp_path / 'tooltip_cameras.yaml'
    front = '/dev/v4l/by-path/pci-usb-0:4:1.0-video-index0'
    side = '/dev/v4l/by-path/pci-usb-0:8:1.0-video-index0'
    save_assignments(path, {'front': front, 'side': side})
    save_measurement(path, 'front', {'device': front}, {'exposure': 420, 'p99': 220, 'saturated_fraction': 0, 'contrast': 100})
    save_assignments(path, {'front': side, 'side': front})
    data = load_settings(path)
    assert data['devices'] == {'front': side, 'side': front}
    assert measurement_for_device(data, front)['exposure'] == 420
    assert len(list((tmp_path / 'history').glob('*.yaml'))) == 2


def test_preview_events_do_not_replace_tooltip_status(tmp_path):
    status = MainStatus(tmp_path)
    status.observe({'event': 'tooltip_start', 'tool': 1})
    status.observe({'event': 'tooltip_preview_error', 'reason': 'USB 带宽不足'})
    assert status.data['tooltip']['event'] == 'tooltip_start'
    assert classify_event({'event': 'tooltip_preview_error'})['color'] == 'red'
    assert classify_event({'event': 'tooltip_exposure_failed'})['color'] == 'red'
    assert classify_event({'event': 'tooltip_preview_open'}) is None
    if status.timer:
        status.timer.cancel()


def test_enumeration_filters_non_capture(tmp_path, monkeypatch):
    import struct
    from nest_clear.devices.calib_cameras import enumerate_capture_devices
    capture = tmp_path / 'a-video-index0'
    metadata = tmp_path / 'b-video-index0'
    ignored = tmp_path / 'c-video-index1'
    for path in (capture, metadata, ignored):
        path.touch()
    def query(fd, request, data, mutate):
        path = __import__('os').readlink(f'/proc/self/fd/{fd}')
        struct.pack_into('II', data, 84, 1 if path == str(capture) else 0, 0)
    monkeypatch.setattr('nest_clear.devices.calib_cameras.fcntl.ioctl', query)
    assert enumerate_capture_devices(tmp_path) == [str(capture)]


def test_enumeration_deduplicates_by_realpath_and_prefers_usb(tmp_path, monkeypatch):
    import struct
    from nest_clear.devices.calib_cameras import enumerate_capture_devices
    targets = tmp_path / 'dev'
    targets.mkdir()
    video0 = targets / 'video0'
    video1 = targets / 'video1'
    video2 = targets / 'video2'
    for video in (video0, video1, video2):
        video.touch()
    by_path = tmp_path / 'by-path'
    by_path.mkdir()
    # Two symlinks point to video0: -usbv2- sorts before -usb- when prefixed with a/b,
    # and the real pci- names have -usb- vs -usbv2- pointing to video1.
    link_usbv2_alpha = by_path / 'a-usbv2-0:8:1.0-video-index0'
    link_usb_alpha = by_path / 'b-usb-0:8:1.0-video-index0'
    link_pci_usb = by_path / 'pci-0000:00:14.0-usb-0:8:1.0-video-index0'
    link_pci_usbv2 = by_path / 'pci-0000:00:14.0-usbv2-0:8:1.0-video-index0'
    link_other = by_path / 'pci-0000:00:14.0-usb-0:4:1.0-video-index0'
    link_usbv2_alpha.symlink_to(video0)
    link_usb_alpha.symlink_to(video0)
    link_pci_usb.symlink_to(video1)
    link_pci_usbv2.symlink_to(video1)
    link_other.symlink_to(video2)

    def query(fd, request, data, mutate):
        struct.pack_into('II', data, 84, 1, 0)

    monkeypatch.setattr('nest_clear.devices.calib_cameras.fcntl.ioctl', query)
    assert enumerate_capture_devices(by_path) == [
        str(link_usb_alpha),
        str(link_other),
        str(link_pci_usb),
    ]


def test_baseline_detects_assignment_and_exposure_change(tmp_path):
    from types import SimpleNamespace
    from nest_clear.devices.tooltip_camera_store import baseline_differences
    path=tmp_path/'tooltip_cameras.yaml'
    a='/dev/v4l/by-path/pci-usb-0:4:1.0-video-index0'
    b='/dev/v4l/by-path/pci-usb-0:8:1.0-video-index0'
    save_assignments(path, {'front':a, 'side':b})
    save_measurement(path, 'front', {'device':a, 'format':'YUYV', 'width':1920, 'height':1080, 'controls':{'exposure':100}}, {'exposure':100,'p99':220,'saturated_fraction':0,'contrast':100})
    cfg=SimpleNamespace(tool_calibration=SimpleNamespace(directory=str(tmp_path), cameras={'front':{'device':a, 'camera_controls':{'brightness':None}}, 'side':{'device':b}}))
    recorded={'front':{'device':a,'format':'YUYV','width':1920,'height':1080,'controls':{'exposure':100}}}
    assert baseline_differences(cfg, recorded)==[]
    save_measurement(path, 'front', {'device':a, 'format':'YUYV', 'width':1920, 'height':1080, 'controls':{'exposure':200}}, {'exposure':200,'p99':220,'saturated_fraction':0,'contrast':100})
    assert 'front.controls.exposure' in baseline_differences(cfg, recorded)
    save_assignments(path, {'front':b,'side':a})
    assert 'front.device' not in baseline_differences(cfg, recorded)  # Legacy devices field is ignored.

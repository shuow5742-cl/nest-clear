"""V4L2 capture device discovery without image-processing dependencies."""
import fcntl
import os
import struct
from pathlib import Path


def _is_capture_device(path: str | Path) -> bool:
    """Return whether a V4L2 device node supports video capture."""
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        cap = bytearray(104)
        fcntl.ioctl(fd, 0x80685600, cap, True)
        capabilities, device_caps = struct.unpack_from('II', cap, 84)
        effective = device_caps if capabilities & 0x80000000 else capabilities
        return bool(effective & (0x1 | 0x1000))
    finally:
        os.close(fd)


def enumerate_capture_devices(directory: str | Path = '/dev/v4l/by-path') -> list[str]:
    """Return by-path index0 nodes which report video capture capability."""
    candidates: dict[str, Path] = {}
    for path in sorted(Path(directory).glob('*-video-index0')):
        real = os.path.realpath(path)
        current = candidates.get(real)
        if current is None or ('-usb-' in path.name and '-usb-' not in current.name):
            candidates[real] = path
    result = []
    for path in sorted(candidates.values()):
        try:
            if _is_capture_device(path):
                result.append(str(path))
        except OSError:
            continue
    return result


def usb_port(path):
    """Return the USB bus and port chain as integers, plus its display label."""
    import re
    match = re.search(r'(?:^|-)usb-(\d+):([0-9]+(?:\.[0-9]+)*)(?=:)', Path(path).name)
    if not match:
        raise ValueError(f'无法识别对刀相机 USB 插口：{path}')
    return (int(match[1]), *(int(part) for part in match[2].split('.'))), 'usb-' + match[1] + ':' + match[2]


def usb_ids(path, sys_class='/sys/class/video4linux'):
    """Read the USB device IDs from the video node's sysfs ancestors."""
    node = Path(os.path.realpath(path)).name
    current = (Path(sys_class) / node / 'device').resolve()
    for parent in (current, *current.parents):
        try:
            vendor = (parent / 'idVendor').read_text().strip().lower()
            product = (parent / 'idProduct').read_text().strip().lower()
            return vendor, product
        except OSError:
            continue
    return None


def resolve_tooltip_roles(vendor='05a3', product='9230', *, directory='/dev/v4l/by-path', sys_class='/sys/class/video4linux'):
    """Require exactly two matching capture devices and assign numeric USB order."""
    found = []
    for path in enumerate_capture_devices(directory):
        if usb_ids(path, sys_class) == (vendor.lower(), product.lower()):
            order, label = usb_port(path)
            found.append((order, label, path))
    found.sort(key=lambda item: item[0])
    labels = '、'.join(item[1] for item in found)
    if not found:
        raise ValueError('未检测到对刀相机')
    if len(found) == 1:
        raise ValueError(f'只检测到 1 台对刀相机（{labels}），需两台都接上才能区分正/侧')
    if len(found) != 2:
        raise ValueError(f'检测到 {len(found)} 台对刀相机（{labels}），需要恰好两台')
    if found[0][0] == found[1][0]:
        raise ValueError(f'对刀相机插口无法区分（{labels}）')
    return {'front': found[0][2], 'side': found[1][2]}

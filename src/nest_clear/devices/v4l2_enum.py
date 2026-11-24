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

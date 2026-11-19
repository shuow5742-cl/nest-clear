"""枚举大恒相机，供现场确认序列号、型号和接口。"""

import sys


def list_cameras() -> list[dict]:
    try:
        import gxipy
    except ImportError as exc:
        raise RuntimeError("无法导入 gxipy：请安装大恒 Galaxy SDK 及其 Python 绑定。") from exc
    _, devices = gxipy.DeviceManager().update_device_list()
    return list(devices)


def main() -> int:
    try:
        devices = list_cameras()
    except Exception as exc:
        print(f"相机枚举失败：{exc}", file=sys.stderr)
        return 1
    if not devices:
        print("未发现大恒相机")
    for device in devices:
        sn = device.get("sn", device.get("serial_number", "未知"))
        model = device.get("model_name", device.get("model", "未知"))
        interface = device.get("device_class", device.get("interface", "未知"))
        print(f"SN: {sn}  型号: {model}  接口: {interface}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

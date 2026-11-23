from nest_clear.guard import GuardError

def resolve_camera():
    from nest_clear.tools.list_cameras import list_cameras
    try:
        devices = list_cameras()
        serials = [str(d.get('sn', d.get('serial_number', ''))) for d in devices]
        if len(serials) != 1 or not serials[0]:
            raise ValueError(f'需要恰好一台带 SN 的相机，发现 {len(serials)} 台：{serials}')
        return serials[0]
    except Exception as exc:
        raise GuardError('CAMERA-AUTO', str(exc), '自动选择相机', '检查大恒 SDK 和相机连接。') from exc


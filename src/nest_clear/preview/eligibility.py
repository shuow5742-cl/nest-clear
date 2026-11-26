"""Camera ownership rules shared by preview clients and hosts."""

from nest_clear.supervisor.device_state import BIT_ESTOP, BIT_MANUAL, BIT_STOP

CAMERAS = ('main', 'front', 'side')
MODE_CAMERAS = {
    '空闲': frozenset(),
    '挑毛中': frozenset({'main'}),
    '相机标定中': frozenset({'main'}),
    '工具偏移标定中': frozenset({'main', 'front', 'side'}),
    '挑取精度标定中': frozenset({'main'}),
    '对刀基准建立中': frozenset({'front', 'side'}),
    '对刀中(工具1)': frozenset({'front', 'side'}),
    '对刀中(工具2)': frozenset({'front', 'side'}),
}
CAMERA_LABELS = {'main': '主相机', 'front': '主对刀相机', 'side': '辅助对刀相机'}


def preview_eligibility(supervisor, age_s, mode, camera):
    """Return (allowed, reason) for a single camera."""
    if camera not in CAMERAS:
        raise ValueError('未知相机')
    if not supervisor.get('pid') or age_s is None or age_s > 3:
        return False, '父程序未运行'
    word = supervisor.get('word')
    if isinstance(word, int) and word & sum(1 << bit for bit in (BIT_STOP, BIT_MANUAL, BIT_ESTOP)):
        return True, None
    running = bool(supervisor.get('child_pid')) or supervisor.get('child_state') in ('启动中', '运行', '就绪')
    if not running:
        return True, None
    # Unknown mode must not silently allow a competing capture.
    used = MODE_CAMERAS.get(mode, frozenset(CAMERAS))
    if camera not in used:
        return True, None
    return False, f'{mode or "主程序状态未知"}使用{CAMERA_LABELS[camera]}，预览不可用'

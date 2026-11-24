"""Synchronous preview handoff before camera-using work."""

from nest_clear.preview.eligibility import MODE_CAMERAS

# Only requests whose branch can capture; mode entry is handled separately.
REQUEST_CAMERAS = {
    20: frozenset({'main'}), 80: frozenset({'main'}),
    95: frozenset({'main'}), 155: frozenset({'main'}),
    100: frozenset({'front', 'side'}), 101: frozenset({'front', 'side'}),
    110: frozenset({'front', 'side'}), 120: frozenset({'front', 'side'}),
}


def close_previews_for(events, cameras, reason, task_id=None, func=None, *, mode=None):
    """Release previews before business captures; main host can supply a stop callback."""
    cameras = set(cameras)
    stopped = []
    preparation = getattr(events, 'tooltip_preparation', None)
    if preparation and cameras.intersection(('front', 'side')):
        stopped.extend(preparation.stop_previews(cameras.intersection(('front', 'side'))))
    stop_main = getattr(events, 'stop_main_preview', None)
    if 'main' in cameras and stop_main and stop_main():
        stopped.append('main')
    for camera in stopped:
        events.emit('preview_auto_close', camera=camera, mode=mode, task_id=task_id,
                    func=func, reason=reason)


def close_for_mode(events, mode, task_id=None, func=None):
    close_previews_for(events, MODE_CAMERAS[mode], 'mode_entry', task_id, func, mode=mode)

"""下针深度标定的独立配置默认值。"""

DEFAULTS = {
    'camera_z_min_mm': None,
    'center_tol_px': 15.,
    'center_max_iter': 5,
    'roi_mm': 8.,
    'coarse_range_mm': 3.,
    'coarse_step_mm': .2,
    'defocus_ratio': .3,
    'fine_range_mm': .6,
    'fine_step_mm': .02,
    'overshoot_mm': .3,
    'repeats_per_point': 3,
    'frames_per_step': 3,
    'metric': 'tenengrad',
    'peak_method': 'gauss',
    'plane_tol_mm': .02,
    'repeat_range_tol_mm': .02,
    'max_points': 9,
    'point_order': None,
    'extra_points': None,
    'settle_samples': 6,
    'settle_window_s': .5,
    'jitter_tol_mm': .02,
    'position_tolerance_mm': .01,
}
METRICS = ('tenengrad', 'laplacian', 'brenner')

"""菲林图案在已知 Z 焦点的合成扫描。"""

import cv2
import numpy as np
import pytest

from nest_clear.depthcal.focus import fit_peak, scores
from nest_clear.depthcal.session import descending


def film_ring(size=160, center=(80, 80)):
    image = np.full((size, size), 210, np.float32)
    x, y = center
    cv2.circle(image, center, 37, 20, 6)
    cv2.line(image, (x - 70, y), (x + 70, y), 20, 2)
    cv2.line(image, (x, y - 70), (x, y + 70), 20, 2)
    return image


@pytest.mark.parametrize(('noise', 'tolerance'), [(0., .002), (2., .01)])
def test_synthetic_focus(noise, tolerance):
    image = film_ring()
    focus = -5.137
    rng = np.random.default_rng(88)
    rows = []
    for z in descending(-4.5, -5.7, .02):
        sigma = max(.0001, 6 * abs(z - focus))
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        metrics = [scores(np.clip(blurred * rng.uniform(.8, 1.15) +
                                  rng.normal(0, noise, image.shape), 0, 255))
                   for _ in range(3)]
        rows.append({'z_actual': z, **{k: float(np.median([m[k] for m in metrics])) for k in metrics[0]}})
    fit = fit_peak(rows, 'tenengrad', 'gauss', '细扫')
    error = abs(fit['z_peak'] - focus)
    print(f'noise={noise}, peak={fit["z_peak"]:.8f}, error={error:.8f}')
    assert error <= tolerance
    assert fit['fwhm'] > 0 and fit['r2'] > .8
    assert fit['half_height'] is not None

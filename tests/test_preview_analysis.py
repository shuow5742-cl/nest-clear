import numpy as np

from nest_clear.preview.analysis import analyze
from test_camcal_core import rendered_board


def test_rendered_board_offset():
    image, _ = rendered_board(5.)
    overlay, result = analyze(image, {})
    assert overlay.shape[:2] == image.shape[:2]
    assert result['corners'] >= 8
    assert np.allclose(result['board_offset_mm'], [5., 0.], atol=.5)

from nest_clear.guard import GuardError
import numpy as np
import pytest
from nest_clear.tooltip import (
    ViewCalib,
    ToolTipBaseline,
    fit_view_jacobian,
    solve_correction,
    is_aligned,
    compute_compensation,
    save_baseline,
    load_baseline,
    ToolCalSchemaError,
    SilhouetteTipDetector,
)


def baseline():
    return ToolTipBaseline(
        "tool",
        (10, 20, 30),
        (
            ViewCalib("a", (100, 100), ((10, 0, 0), (0, 0, 10))),
            ViewCalib("b", (200, 200), ((0, 10, 0), (0, 0, 10))),
        ),
        (1, 2, 3),
    )


def test_random_fit_and_solve():
    rng = np.random.default_rng(42)
    js = rng.normal(size=(2, 2, 3)) * 30
    x = rng.normal(size=(100, 3))
    views = []
    for i, j in enumerate(js):
        y = x @ j.T + rng.normal(scale=0.001, size=(100, 2))
        fit = fit_view_jacobian(zip(x, y))
        assert np.max(np.abs(fit.jacobian - j)) < 0.001
        assert fit.rms_px < 0.002
        views.append(ViewCalib(str(i), (10, 20), fit.jacobian))
    delta = np.array([0.1, -0.2, 0.3])
    observed = {
        v.name: np.asarray(v.target_uv) - j @ delta + rng.normal(scale=0.001, size=2)
        for v, j in zip(views, js)
    }
    result = solve_correction(observed, views)
    np.testing.assert_allclose(result.delta_xyz, delta, atol=0.001)
    assert not is_aligned(result, 0.01)
    assert is_aligned(solve_correction({v.name: v.target_uv for v in views}, views), 0.001)


def test_degenerate_and_missing():
    with pytest.raises(GuardError, match="three"):
        fit_view_jacobian([((1, 0, 0), (1, 1))])
    with pytest.raises(GuardError, match="rank deficient"):
        fit_view_jacobian([((i, 0, 0), (i, i)) for i in range(4)])
    v = baseline().views[0]
    with pytest.raises(GuardError, match="underdetermined"):
        solve_correction({"a": (100, 100)}, [v])
    with pytest.raises(GuardError, match="missing"):
        solve_correction({"a": (100, 100)}, baseline())


def test_disagreement_and_compensation():
    b = baseline()
    result = solve_correction({"a": (100, 90), "b": (200, 180)}, b)
    assert result.axis_disagreement["z"] == pytest.approx(1)
    c = compute_compensation(b, (11, 18, 33))
    np.testing.assert_allclose(c.delta_xyz, (1, -2, 3))
    np.testing.assert_allclose(c.offset_xyz, (2, 0, 6))
    np.testing.assert_allclose(
        compute_compensation(b, (11, 18, 33), convention="target_minus_compensation").offset_xyz,
        (0, 4, 0),
    )


def test_yaml(tmp_path):
    path = tmp_path / "baseline.yaml"
    save_baseline(baseline(), path)
    assert load_baseline(path) == baseline()
    path.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ToolCalSchemaError, match="duplicate"):
        load_baseline(path)
    path.write_text("version: 2\n")
    with pytest.raises(ToolCalSchemaError, match="missing"):
        load_baseline(path)


def test_detector():
    image = np.full((100, 100), 255, np.uint8)
    image[20:81, 47:54] = 0
    detector = SilhouetteTipDetector(direction=(0, 1))
    assert np.linalg.norm(np.array(detector.detect(image).point_uv) - (50, 80)) < 1
    assert detector.detect(np.full_like(image, 255)).point_uv is None

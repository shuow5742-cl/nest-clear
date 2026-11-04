from nest_clear.guard import GuardError
import numpy as np
import pytest
import yaml

from nest_clear.tooltip import (
    SilhouetteTipDetector,
    ToolCalSchemaError,
    ToolTipBaseline,
    ViewCalib,
    check_quality,
    compute_compensation,
    fit_view_jacobian,
    is_aligned,
    load_baseline,
    save_baseline,
    solve_correction,
)


@pytest.fixture
def base():
    return ToolTipBaseline(
        "tool",
        (1, 2, 3),
        (
            ViewCalib("a", (100, 100), ((10, 0, 0), (0, 0, 10))),
            ViewCalib("b", (200, 200), ((0, 10, 0), (0, 0, 10))),
        ),
        (4, 5, 6),
        "2026-01-01",
        "reference",
    )


@pytest.mark.parametrize("seed", range(12))
def test_random_noisy_roundtrip(seed):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(2, 2, 3)) * 40
    while np.linalg.cond(j.reshape(4, 3)) > 8:
        j = rng.normal(size=(2, 2, 3)) * 40
    x = rng.normal(size=(80, 3))
    delta = rng.uniform(-0.5, 0.5, size=3)
    views = []
    observed = {}
    for i in range(2):
        y = x @ j[i].T + rng.normal(scale=0.01, size=(80, 2))
        fit = fit_view_jacobian(zip(x, y))
        assert np.max(np.abs(fit.jacobian - j[i])) < 0.006
        view = ViewCalib(str(i), (100, 200), fit.jacobian)
        views.append(view)
        observed[view.name] = (
            np.array(view.target_uv) - j[i] @ delta + rng.normal(scale=0.01, size=2)
        )
    solved = solve_correction(observed, views)
    assert np.max(np.abs(solved.delta_xyz - delta)) < 0.003


def test_fit_residual_statistics():
    x = np.vstack((np.eye(3), -np.eye(3)))
    j = np.array([[2, 3, 4], [5, 6, 7]])
    y = x @ j.T + np.array([[0.1, -0.2]] * 6)
    fit = fit_view_jacobian(zip(x, y))
    np.testing.assert_allclose(fit.jacobian, j, atol=1e-12)
    np.testing.assert_allclose(fit.residuals_uv, np.tile([0.1, -0.2], (6, 1)), atol=1e-12)
    assert fit.rms_px == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))
    assert fit.condition_number == pytest.approx(1)


def test_insufficient_samples():
    with pytest.raises(GuardError, match="at least three independent"):
        fit_view_jacobian([((1, 0, 0), (1, 2))] * 2)


def test_collinear_samples():
    with pytest.raises(GuardError, match="rank deficient.*three axes"):
        fit_view_jacobian([((i, 2 * i, 3 * i), (i, i)) for i in range(1, 5)])


def test_rank_two_samples():
    with pytest.raises(GuardError, match="rank deficient.*three axes"):
        fit_view_jacobian([((1, 0, 0), (1, 0)), ((0, 1, 0), (0, 1)), ((1, 1, 0), (1, 1))])


def test_single_view_underdetermined(base):
    with pytest.raises(GuardError, match="underdetermined"):
        solve_correction({"a": (100, 100)}, base.views[:1])


def test_missing_view(base):
    with pytest.raises(GuardError, match="missing.*b"):
        solve_correction({"a": (100, 100)}, base)


def test_weights_and_primary_view(base):
    obs = {"a": (100, 90), "b": (200, 180)}
    unweighted = solve_correction(obs, base)
    weighted = solve_correction(obs, base, weights={"a": 3})
    primary = solve_correction(obs, base, primary_view="a")
    assert unweighted.delta_xyz[2] == pytest.approx(1.5)
    assert weighted.delta_xyz[2] == pytest.approx(1.25)
    assert primary.delta_xyz[2] == pytest.approx(4 / 3)


def test_axis_disagreement(base):
    result = solve_correction({"a": (100, 90), "b": (200, 180)}, base)
    assert result.axis_estimates["z"] == pytest.approx({"a": 1, "b": 2})
    assert result.axis_disagreement["z"] == pytest.approx(1)
    assert "x" not in result.axis_disagreement


def test_condition_number_grows():
    def solve(eps):
        views = [
            ViewCalib("a", (0, 0), ((1, 0, 0), (0, 1, 0))),
            ViewCalib("b", (0, 0), ((1, 0, 0), (0, 0, eps))),
        ]
        return solve_correction({"a": (0, 0), "b": (0, 0)}, views)

    assert solve(0.001).condition_number > 100 * solve(1).condition_number


def test_alignment_independent_of_quality(base):
    result = solve_correction({"a": (100, 99), "b": (200, 201)}, base)
    assert is_aligned(result, 0.001)
    quality = check_quality(result, 0.05, 100)
    assert not quality.ok and not quality.residual_ok and quality.condition_ok
    assert quality.residuals_mm == pytest.approx({"a": 0.1, "b": 0.1})


def test_alignment_axis_tolerance(base):
    obs = {"a": (99.1, 100), "b": (199.1, 200)}
    result = solve_correction(obs, base)
    assert np.linalg.norm(result.delta_xyz) > 0.1
    assert is_aligned(result, 0.1)
    assert not is_aligned(result, 0.08)


def test_quality_condition_failure():
    views = [
        ViewCalib("a", (0, 0), ((1, 0, 0), (0, 1, 0))),
        ViewCalib("b", (0, 0), ((1, 0, 0), (0, 0, 0.001))),
    ]
    result = solve_correction({"a": (0, 0), "b": (0, 0)}, views)
    quality = check_quality(result, 0.01, 100)
    assert quality.residual_ok and not quality.condition_ok and not quality.ok


def test_compensation_plus(base):
    c = compute_compensation(base, (2, 0, 6))
    np.testing.assert_allclose(c.delta_xyz, (1, -2, 3))
    np.testing.assert_allclose(c.offset_xyz, (5, 3, 9))


def test_compensation_minus(base):
    c = compute_compensation(base, (2, 0, 6), convention="target_minus_compensation")
    np.testing.assert_allclose(c.offset_xyz, (3, 7, 3))


def test_compensation_invalid_convention(base):
    with pytest.raises(GuardError, match="convention"):
        compute_compensation(base, base.p0, convention="invalid")


def test_yaml_roundtrip(base, tmp_path):
    path = tmp_path / "base.yaml"
    save_baseline(base, path)
    assert load_baseline(path) == base


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "version must be 1"),
        ("p0", None, "missing fields.*p0"),
        ("offset_xyz", [1, 2], "offset_xyz.*shape"),
        ("p0", [1, float("nan"), 3], "p0.*finite"),
        (
            "views",
            [{"name": "a", "target_uv": [1, 2], "jacobian": [[1, 2], [3, 4]]}],
            "jacobian.*shape",
        ),
    ],
)
def test_yaml_invalid_fields(base, tmp_path, field, value, message):
    path = tmp_path / "base.yaml"
    save_baseline(base, path)
    raw = yaml.safe_load(path.read_text())
    if value is None:
        del raw[field]
    else:
        raw[field] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ToolCalSchemaError, match=message):
        load_baseline(path)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ((0, 1), (50, 80)),
        ((0, -1), (50, 20)),
        ((1, 0), (53, 50)),
        ((-1, 0), (47, 50)),
    ],
)
def test_detector_four_directions(direction, expected):
    image = np.full((100, 100), 255, np.uint8)
    image[20:81, 47:54] = 0
    point = SilhouetteTipDetector(direction=direction).detect(image).point_uv
    assert np.linalg.norm(np.array(point) - expected) < 1


def test_detector_no_target():
    image = np.full((100, 100), 255, np.uint8)
    assert SilhouetteTipDetector().detect(image).point_uv is None


def test_detector_largest_component():
    image = np.full((100, 100), 255, np.uint8)
    image[20:81, 47:54] = 0
    image[85:95, 10:12] = 0
    assert SilhouetteTipDetector().detect(image).point_uv == (50, 80)


def test_detector_intensity_and_isolated_noise():
    rng = np.random.default_rng(7)
    image = rng.integers(210, 256, (100, 100), dtype=np.uint8)
    image[20:81, 47:54] = rng.integers(0, 40, (61, 7), dtype=np.uint8)
    image[90, 10] = image[95, 80] = 0
    assert np.linalg.norm(np.array(SilhouetteTipDetector().detect(image).point_uv) - (50, 80)) < 1

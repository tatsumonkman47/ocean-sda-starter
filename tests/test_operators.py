# SPDX-License-Identifier: Apache-2.0
"""General observation operators and correlated observation errors."""

import numpy as np
import pytest
import torch

from obottle_starter.core.normalization import Normalizer
from obottle_starter.inference.general import assimilate_general, model_unit_observation
from obottle_starter.inference.sda.mmps import MMPSDenoiser
from obottle_starter.inference.sda.operators import (
    ObservationCovariance,
    ObservationSet,
    block_average,
    exponential_covariance,
    footprint_observations,
    gaussian_kernel,
    point_observations,
)
from obottle_starter.inference.sda.rozet import RozetDenoiser

from conftest import tiny_config
from test_mmps import S2, GaussianPriorDenoiser

SHAPE = (2, 3, 8, 8)  # C, T, H, W


def _linear_field(shape=SHAPE):
    c, t, r, w = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return torch.from_numpy(
        (1.0 + 2 * c + 0.5 * t + 0.3 * r - 0.2 * w).astype(np.float32)
    )


def test_points_interpolate_linear_fields_exactly():
    x = _linear_field().reshape(2, 3, 64)
    rng = np.random.default_rng(0)
    f, r, c = rng.uniform(0, 2, 20), rng.uniform(0, 7, 20), rng.uniform(0, 7, 20)
    op = point_observations(SHAPE, 1, f, r, c)
    expected = 1.0 + 2 * 1 + 0.5 * f + 0.3 * r - 0.2 * c
    np.testing.assert_allclose(op(x).numpy(), expected, rtol=1e-5)
    # Combination of variables at the same points.
    combo = point_observations(SHAPE, {0: 2.0, 1: -1.0}, f, r, c)
    np.testing.assert_allclose(
        combo(x).numpy(), 2 * (expected - 2) - expected, rtol=1e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="row positions"):
        point_observations(SHAPE, 0, [0], [7.5], [0])


def test_block_average_and_footprints():
    x = torch.randn(2, 3, 64)
    op = block_average(SHAPE, 0, 4)
    assert op.n_obs == 3 * 2 * 2
    manual = x[0].reshape(3, 2, 4, 2, 4).mean(dim=(2, 4)).reshape(-1)
    torch.testing.assert_close(op(x), manual)
    assert op.coords["row"].tolist()[:2] == [1.5, 1.5]
    # A footprint kernel at the corner still averages a constant to itself.
    const = torch.full((2, 3, 64), 4.0)
    fp = footprint_observations(SHAPE, 1, [0, 2], [0, 4], [0, 4], gaussian_kernel(1.0))
    torch.testing.assert_close(fp(const), torch.full((2,), 4.0))


def test_sums_and_scaling_make_time_averages():
    x = torch.randn(2, 3, 64)
    rows, cols = np.array([1.0, 6.0]), np.array([2.0, 3.5])
    a = point_observations(SHAPE, 0, [0, 0], rows, cols)
    b = point_observations(SHAPE, 0, [2, 2], rows, cols)
    mean = 0.5 * (a + b)
    torch.testing.assert_close(mean(x), 0.5 * (a(x) + b(x)))
    assert mean.footprint(2)[0, 1].sum() == 0  # frame 1 unused


def _random_spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    return m @ m.T / n + 0.1 * torch.eye(n, dtype=torch.float64)


def test_covariance_algebra_matches_dense():
    blocks = [torch.rand(3, dtype=torch.float64) + 0.5, _random_spd(4, 1)]
    cov = ObservationCovariance(blocks)
    dense = torch.block_diag(torch.diag(blocks[0]), blocks[1])
    v = torch.randn(2, 7, dtype=torch.float64)
    torch.testing.assert_close(cov.matvec(v), v @ dense.T)
    torch.testing.assert_close(cov.solve(v), torch.linalg.solve(dense, v.T).T)
    w = cov.whiten(v)  # |L⁻¹ v|² = vᵀ R⁻¹ v
    torch.testing.assert_close(
        w.square().sum(-1), (v * torch.linalg.solve(dense, v.T).T).sum(-1)
    )
    torch.testing.assert_close(
        cov.shifted(0.3).matvec(v),
        v @ (dense + 0.3 * torch.eye(7, dtype=torch.float64)).T,
    )
    q = torch.rand(7, dtype=torch.float64) + 0.5
    torch.testing.assert_close(
        cov.scaled(q).matvec(v), v @ (q[:, None] * dense * q[None, :]).T
    )
    draws = torch.stack(
        [cov.sample(torch.Generator().manual_seed(i)) for i in range(4000)]
    )
    torch.testing.assert_close(torch.cov(draws.T), dense, atol=0.08, rtol=0)


def _mixed_problem():
    shape = (2, 2, 6, 6)
    rng = np.random.default_rng(3)
    pts = point_observations(
        shape,
        {0: 1.0, 1: 0.5},
        rng.uniform(0, 1, 5),
        rng.uniform(0, 5, 5),
        rng.uniform(0, 5, 5),
        name="track",
    )
    pts.offset = torch.randn(5)
    coarse = block_average(shape, 1, 3, name="coarse")
    op = ObservationSet([pts, coarse])
    cov = ObservationCovariance(
        [
            exponential_covariance(np.arange(5.0), 0.05, 2.0, nugget=0.01),
            torch.full((coarse.n_obs,), 0.03**2),
        ]
    )
    return op, cov, (2, 2, 36)


def _dense_A(op, shape):
    n = int(np.prod(shape))
    eye = torch.eye(n, dtype=torch.float64).reshape(n, *shape)
    offset = op(torch.zeros(1, *shape, dtype=torch.float64))[0]
    return (op(eye) - offset).T, offset


@pytest.mark.parametrize("solver", ["cg", "gmres", "minres"])
@pytest.mark.parametrize("preconditioner", ["none", "diagonal"])
def test_mmps_exact_with_general_operators_and_correlated_errors(
    solver, preconditioner
):
    torch.set_default_dtype(torch.float64)
    try:
        op, cov, shape = _mixed_problem()
        cov = ObservationCovariance([b.double() for b in cov.blocks])
        A, offset = _dense_A(op, shape)
        R = torch.block_diag(*[b if b.ndim == 2 else torch.diag(b) for b in cov.blocks])
        y = torch.randn(op.n_obs)
        guided = MMPSDenoiser(
            GaussianPriorDenoiser(),
            y,
            op,
            cov,
            solver=solver,
            iterations=op.n_obs,
            preconditioner=preconditioner,
        )
        x_t = torch.randn(2, *shape)
        t = torch.tensor(0.4)
        sigma = float(guided.schedule(t)[1])
        m = S2 / (S2 + sigma**2) * x_t.reshape(2, -1)
        P = S2 * sigma**2 / (S2 + sigma**2)
        K = P * A.T @ torch.linalg.inv(P * A @ A.T + R)
        exact = (m + (y - (m @ A.T + offset)) @ K.T).reshape(x_t.shape)
        torch.testing.assert_close(guided(x_t, t).mean, exact, rtol=1e-6, atol=1e-7)
    finally:
        torch.set_default_dtype(torch.float32)


def test_rozet_with_correlated_errors_follows_its_formula():
    torch.set_default_dtype(torch.float64)
    try:
        op, cov, shape = _mixed_problem()
        cov = ObservationCovariance([b.double() for b in cov.blocks])
        A, offset = _dense_A(op, shape)
        R = torch.block_diag(*[b if b.ndim == 2 else torch.diag(b) for b in cov.blocks])
        y = torch.randn(op.n_obs)
        guided = RozetDenoiser(GaussianPriorDenoiser(), y, op, cov, gamma=0.1)
        x_t = torch.randn(2, *shape)
        t = torch.tensor(0.3)
        sigma = float(guided.schedule(t)[1])
        s = S2 / (S2 + sigma**2)
        m = s * x_t.reshape(2, -1)
        resid = y - (m @ A.T + offset)
        g = torch.linalg.solve(R + 0.1 * sigma**2 * torch.eye(op.n_obs), resid.T).T
        exact = (m + sigma**2 * s * g @ A).reshape(x_t.shape)
        torch.testing.assert_close(guided(x_t, t).mean, exact, rtol=1e-6, atol=1e-8)
    finally:
        torch.set_default_dtype(torch.float32)


def test_plain_variance_tensors_behave_exactly_as_before():
    op, _, shape = _mixed_problem()
    var = torch.full((op.n_obs,), 0.02**2)
    x_t = torch.randn(2, *shape)
    t = torch.tensor(0.5)
    y = torch.randn(op.n_obs)
    for cls, kw in (
        (MMPSDenoiser, {"iterations": 3, "cov_y_inflation": 0.1}),
        (RozetDenoiser, {"gamma": 0.1}),
    ):
        a = cls(GaussianPriorDenoiser(), y, op, var, **kw)(x_t, t).mean
        b = cls(GaussianPriorDenoiser(), y, op, ObservationCovariance([var]), **kw)(
            x_t, t
        ).mean
        assert torch.equal(a, b)


# ── Physical units ────────────────────────────────────────────────────────

CFG = tiny_config()
TIMES = np.datetime64("2012-03-01T00") + np.arange(3) * np.timedelta64(12, "h")


def _physical_problem():
    shape = (2, 3, 16, 16)
    rng = np.random.default_rng(0)
    truth = {"SSH": rng.normal(0, 0.05, shape[1:]), "SST": rng.normal(18, 1, shape[1:])}
    sst_pts = point_observations(
        shape,
        1,
        rng.uniform(0, 2, 30),
        rng.uniform(0, 15, 30),
        rng.uniform(0, 15, 30),
        name="SST_points",
    )
    ssh_coarse = block_average(shape, 0, 4, name="SSH_coarse")
    op = ObservationSet([sst_pts, ssh_coarse])
    covs = [
        exponential_covariance(np.arange(30.0), 0.1, 5.0, nugget=0.02),
        torch.full((ssh_coarse.n_obs,), 0.01**2),
    ]
    return op, covs, truth


def test_model_unit_observation_is_consistent():
    norm = Normalizer.from_config(CFG)
    op, covs, truth = _physical_problem()
    spec = model_unit_observation(op, covs, norm, TIMES, truth=truth)
    q = spec.extras["scaling"]
    # Default Q = 1 / error std: every observation in units of its error.
    sd = np.sqrt(0.1**2 + 0.02**2)
    torch.testing.assert_close(q[:30], torch.full((30,), 1 / sd))
    torch.testing.assert_close(q[30:], torch.full((spec.y.numel() - 30,), 1 / 0.01))
    torch.testing.assert_close(spec.cov_y.diagonal(), torch.ones(spec.y.numel()))
    # The model-unit operator on the normalized truth reproduces Q y_phys.
    torch.testing.assert_close(spec.operator(spec.truth), spec.y, rtol=1e-4, atol=1e-3)
    # "row_norm": Q = 1 / scale for single-variable points and averages, and
    # R_model = Q R Q.
    spec = model_unit_observation(
        op, covs, norm, TIMES, truth=truth, scaling="row_norm"
    )
    q = spec.extras["scaling"]
    torch.testing.assert_close(q[:30], torch.full((30,), 1 / 5.5))
    torch.testing.assert_close(q[30:], torch.full((spec.y.numel() - 30,), 1 / 0.051))
    torch.testing.assert_close(spec.operator(spec.truth), spec.y, rtol=1e-4, atol=1e-4)
    R0 = covs[0].float()
    torch.testing.assert_close(spec.cov_y.blocks[0], R0 / 5.5**2)


def test_assimilate_general_end_to_end(tiny_net, tiny_cfg):
    op, covs, truth = _physical_problem()
    ds = assimilate_general(
        tiny_net,
        tiny_cfg,
        op,
        covs,
        times=TIMES,
        center_lon=-40.0,
        truth=truth,
        instrument_noise=True,
        n_members=2,
        steps=3,
        verbose=False,
        seed=0,
    )
    assert ds.samples.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(ds.samples).all() and np.isfinite(ds.misfit).all()
    assert ds.predicted.shape == (2, op.n_obs) and "obs_row" in ds
    assert (
        ds.attrs["correlated_errors"] == 1
        and ds.attrs["groups"] == "SST_points,SSH_coarse"
    )
    np.testing.assert_allclose(ds.truth.sel(channel="SST"), truth["SST"], rtol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        assimilate_general(
            tiny_net,
            tiny_cfg,
            block_average((2, 3, 8, 8), 0, 4),
            0.01,
            times=TIMES,
            center_lon=0.0,
            truth=truth,
            verbose=False,
        )


def test_invalid_observations_and_covariances_are_rejected():
    norm = Normalizer.from_config(CFG)
    op, covs, truth = _physical_problem()
    bad_y = np.zeros(op.n_obs)
    bad_y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model_unit_observation(op, covs, norm, TIMES, y=bad_y)
    for bad, match in [
        (torch.full((op.parts[1].n_obs,), float("inf")), "NaN or inf"),
        (torch.zeros(op.parts[1].n_obs), "<= 0"),
    ]:
        with pytest.raises(ValueError, match=match):
            model_unit_observation(op, [covs[0], bad], norm, TIMES, truth=truth)
    not_pd = torch.ones(30, 30, dtype=torch.float64)  # rank one
    with pytest.raises(ValueError, match="positive definite"):
        model_unit_observation(op, [not_pd, covs[1]], norm, TIMES, truth=truth)
    # NaN in the truth only matters where it is observed.
    holey = {k: v.copy() for k, v in truth.items()}
    holey["SSH"][0, 0, 0] = np.nan  # inside the first SSH block average
    with pytest.raises(ValueError, match="finite"):
        model_unit_observation(op, covs, norm, TIMES, truth=holey)

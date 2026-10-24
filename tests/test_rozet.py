# SPDX-License-Identifier: Apache-2.0
"""Rozet & Louppe (2023) guidance on a Gaussian prior.

For ``x ~ N(0, s² I)`` the denoiser is ``x̂ = a x_t`` with ``a = s²/(s²+σ²)``
and the true ``Cov[x | x_t] = a σ² I``. Rozet's heuristic replaces that with
``γ σ² I``, so with ``γ = a`` and unit-weight masks it is exact.
"""

import pytest
import torch

from obottle_starter.inference.sda.observations import (
    ObservationOperator,
    ObservationTerm,
)
from obottle_starter.inference.sda.rozet import RozetDenoiser

from test_mmps import S2, GaussianPriorDenoiser, _exact, _operator


def _matrix(op, shape):
    n = int(torch.tensor(shape).prod())
    eye = torch.eye(n, dtype=torch.float64).reshape(n, *shape)
    offset = op(torch.zeros(1, *shape, dtype=torch.float64))[0]
    return (op(eye) - offset).T, offset


def test_matches_closed_form():
    torch.set_default_dtype(torch.float64)
    try:
        op, shape = _operator()  # includes a two-channel term and offsets
        n_obs = sum(op.sizes)
        y, cov_y, gamma = torch.randn(n_obs), torch.full((n_obs,), 0.02**2), 0.3
        guided = RozetDenoiser(GaussianPriorDenoiser(), y, op, cov_y, gamma=gamma)
        x_t, t = torch.randn(2, *shape), torch.tensor(0.4)
        sigma = float(guided.schedule(t)[1])
        a = S2 / (S2 + sigma**2)
        A, offset = _matrix(op, shape)
        x_hat = a * x_t.reshape(2, -1)
        g = (y - (x_hat @ A.T + offset)) / (cov_y + gamma * sigma**2)
        expected = (x_hat + sigma**2 * a * g @ A).reshape(x_t.shape)
        torch.testing.assert_close(guided(x_t, t).mean, expected)
    finally:
        torch.set_default_dtype(torch.float32)


def test_exact_when_gamma_matches_prior():
    torch.set_default_dtype(torch.float64)
    try:
        gen = torch.Generator().manual_seed(1)
        shape = (2, 3, 9)
        terms = [
            ObservationTerm(f"ch{c}", torch.rand(3, 9, generator=gen) < 0.5, {c: 1.0})
            for c in range(2)
        ]
        op = ObservationOperator(terms)
        n_obs = sum(op.sizes)
        y, cov_y = torch.randn(n_obs), torch.full((n_obs,), 0.05**2)
        t = torch.tensor(0.4)
        sigma = float(GaussianPriorDenoiser().schedule(t)[1])
        gamma = S2 / (S2 + sigma**2)
        guided = RozetDenoiser(GaussianPriorDenoiser(), y, op, cov_y, gamma=gamma)
        x_t = torch.randn(2, *shape)
        torch.testing.assert_close(
            guided(x_t, t).mean, _exact(op, shape, x_t, sigma, y, cov_y)
        )
    finally:
        torch.set_default_dtype(torch.float32)


def test_diagnostics_and_bad_gamma():
    op, shape = _operator()
    n_obs = sum(op.sizes)
    guided = RozetDenoiser(
        GaussianPriorDenoiser(),
        torch.randn(n_obs),
        op,
        torch.full((n_obs,), 0.1),
        record_diagnostics=True,
        obs_groups=op.groups,
        group_names=op.names,
    )
    guided(torch.randn(1, *shape), torch.tensor(0.5))
    assert {"misfit_rms", "misfit_rms_SSH", "misfit_rms_SST"} <= set(
        guided.diagnostics[0]
    )
    with pytest.raises(ValueError, match="gamma"):
        RozetDenoiser(
            GaussianPriorDenoiser(), torch.zeros(1), op, torch.ones(1), gamma=-1
        )

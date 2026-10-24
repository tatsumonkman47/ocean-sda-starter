# SPDX-License-Identifier: Apache-2.0
"""MMPS is exact for a Gaussian prior when the linear solve converges.

For ``x ~ N(0, s² I)`` and ``x_t = x + σ ε`` the Tweedie denoiser is
``x̂ = s²/(s²+σ²) x_t`` with covariance ``P = s²σ²/(s²+σ²) I``, so the exact
posterior mean given ``y = A x + noise`` is
``x̂ + P Aᵀ (A P Aᵀ + Σ_y)⁻¹ (y − A x̂)``.
"""

import pytest
import torch

from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import VESchedule
from obottle_starter.inference.sda.mmps import MMPSDenoiser
from obottle_starter.inference.sda.observations import (
    ObservationOperator,
    ObservationTerm,
)

S2 = 0.7  # prior variance


class GaussianPriorDenoiser(Denoiser):
    def __init__(self):
        super().__init__()
        self.schedule = VESchedule(1e-3, 1e2)

    def forward(self, x_t, t, **kwargs):
        _, sigma = self.schedule(t)
        return DiracPosterior(mean=S2 / (S2 + sigma**2) * x_t)


def _operator():
    gen = torch.Generator().manual_seed(0)
    C, T, HW = 3, 2, 9
    ssh = torch.rand(T, HW, generator=gen) < 0.5
    sst = torch.rand(T, HW, generator=gen) < 0.4
    terms = [
        ObservationTerm("SSH", ssh, {0: 0.05, 1: 0.08}, 0.01),
        ObservationTerm("SST", sst, {2: 1.5}, offset=torch.randn(T, HW, generator=gen)),
    ]
    return ObservationOperator(terms), (C, T, HW)


def _exact(op, shape, x_t, sigma, y, cov_y):
    n = x_t[0].numel()
    eye = torch.eye(n, dtype=torch.float64).reshape(n, *shape)
    offset = op(torch.zeros(1, *shape, dtype=torch.float64))[0]
    A = (op(eye) - offset).T
    m = S2 / (S2 + sigma**2) * x_t.reshape(len(x_t), -1)
    P = S2 * sigma**2 / (S2 + sigma**2)
    K = P * A.T @ torch.linalg.inv(P * A @ A.T + torch.diag(cov_y))
    resid = y - (m @ A.T + offset)
    return (m + resid @ K.T).reshape(x_t.shape)


@pytest.mark.parametrize("solver", ["cg", "gmres", "minres"])
@pytest.mark.parametrize("preconditioner", ["none", "diagonal"])
def test_mmps_matches_exact_gaussian_posterior(solver, preconditioner):
    torch.set_default_dtype(torch.float64)
    try:
        op, shape = _operator()
        n_obs = sum(op.sizes)
        y = torch.randn(n_obs)
        cov_y = torch.full((n_obs,), 0.02**2)
        guided = MMPSDenoiser(
            GaussianPriorDenoiser(),
            y,
            op,
            cov_y,
            solver=solver,
            iterations=n_obs,
            preconditioner=preconditioner,
        )
        x_t = torch.randn(2, *shape)
        t = torch.tensor(0.4)
        sigma = float(guided.schedule(t)[1])
        out = guided(x_t, t).mean
        torch.testing.assert_close(
            out, _exact(op, shape, x_t, sigma, y, cov_y), rtol=1e-6, atol=1e-7
        )
    finally:
        torch.set_default_dtype(torch.float32)


def test_bad_options_are_rejected():
    op, _ = _operator()
    args = (GaussianPriorDenoiser(), torch.zeros(1), op, torch.ones(1))
    with pytest.raises(ValueError, match="preconditioner"):
        MMPSDenoiser(*args, preconditioner="ilu")
    with pytest.raises(ValueError, match="solver"):
        MMPSDenoiser(*args, solver="lsqr")

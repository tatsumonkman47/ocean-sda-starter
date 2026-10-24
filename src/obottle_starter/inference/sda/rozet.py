# SPDX-License-Identifier: Apache-2.0
r"""Rozet & Louppe (2023) guidance: a fixed-variance Gaussian likelihood.

    Rozet & Louppe (2023), "Score-based Data Assimilation",
    https://arxiv.org/abs/2306.10574.

The likelihood of the observations given the current noisy state is
approximated as

.. math::
    p(y \mid x_t) \approx \mathcal{N}\big(y \mid A(\hat x),\;
        \Sigma_y + \gamma \, (\sigma_t / \alpha_t)^2 I\big),
    \qquad \hat x = \mu_\phi(x_t),

i.e. ``Cov[x | x_t]`` is replaced by ``γ σ_t² I`` (``α_t = 1`` for the VE
schedule used here). The guided mean is

.. math::
    \hat x + \frac{\sigma_t^2}{\alpha_t} \, J^\top A^\top
        \frac{y - A(\hat x)}{\Sigma_y + \gamma (\sigma_t/\alpha_t)^2},

with ``J = ∂x̂/∂x_t``. Compared with :class:`~obottle_starter.inference.sda.mmps.MMPSDenoiser`
there is no linear solve: each step costs one network forward and one
backward pass, but the variance does not adapt to where the model is
certain, and ``γ`` must be chosen.

The variance term ``γ σ_t²`` is added directly in observation space, one
scalar per observation. Replacing ``Cov[x | x_t]`` by ``γ σ_t² I`` in state
space and propagating it through a general linear operator would instead
give ``γ σ_t² A Aᵀ``. The two coincide when every row of ``A`` selects one
state value with unit weight (plain masks in model units), but not for
smoothing, averaging or weighted combinations of channels. This class
implements the observation-space heuristic, as Rozet & Louppe's code does.

Choosing ``γ``. For unit-weight observations in model units, MMPS's
``A Cov[x | x_t] Aᵀ`` is close to ``σ_t² I`` at small ``σ_t`` (the
denoiser's Jacobian is near the identity) and close to ``σ_data² I`` at
large ``σ_t``. So ``γ = 1`` matches MMPS at the end of sampling and makes
guidance weaker than MMPS early on; smaller ``γ`` strengthens guidance at
every step. Rozet & Louppe's code uses ``γ = 1e-2``. The likelihood
variance is that of ``y``, so ``γ`` assumes observations in model units
(rows of ``A`` of order one), as :func:`obottle_starter.assimilate` builds
them.

The formula follows ``GaussianScore`` in Rozet's ``sda`` code
(https://github.com/francois-rozet/sda), written here as a denoiser so it
runs on the same DDIM sampler as MMPS.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import Schedule

__all__ = ["RozetDenoiser"]


class RozetDenoiser(Denoiser):
    r"""Rozet & Louppe (2023) guidance around an unconditional denoiser.

    Args:
        denoiser: The unconditional denoiser :math:`q_\phi(X \mid X_t)`.
        y: Observations, shape ``(N,)`` or ``(B, N)``.
        A: Differentiable forward operator, ``(B, *x) -> (B, N)``.
        cov_y: Observation noise variances (diagonal of :math:`\Sigma_y`),
            shape ``(N,)`` or broadcastable to ``y``.
        gamma: ``γ`` in the likelihood variance ``Σ_y + γ (σ_t/α_t)²``.
        record_diagnostics: Record the misfit at each step in
            :attr:`diagnostics`. Costs nothing extra.
        obs_groups: Optional ``(N,)`` integer group index of each observation.
        group_names: Names for the groups in ``obs_groups``.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov_y: Tensor,
        *,
        gamma: float = 1e-2,
        record_diagnostics: bool = False,
        obs_groups: Tensor | None = None,
        group_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.denoiser = denoiser
        self.y = y
        self.A = A
        self.cov_y = cov_y
        self.gamma = float(gamma)
        self.record_diagnostics = bool(record_diagnostics)
        self.obs_groups = obs_groups
        self.group_names = list(group_names) if group_names is not None else None
        self.diagnostics: list[dict] = []

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def reset(self) -> None:
        """Forget recorded diagnostics."""
        self.diagnostics = []

    @torch.no_grad()
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, t, **kwargs).mean
            y_hat = self.A(x_hat)

        # Observation-space variance: Σ_y + γ (σ_t/α_t)², one value per
        # observation (see the module docstring for when this equals the
        # state-space heuristic propagated through A).
        var = self.cov_y + self.gamma * (sigma_t / alpha_t) ** 2
        b = self.y - y_hat
        g = b / var
        # σ_t²/α_t · Jᵀ Aᵀ g: autograd.grad gives the vector-Jacobian
        # product through A and the network (Jᵀ v, not J v).
        correction = sigma_t**2 / alpha_t * torch.autograd.grad(y_hat, x_t, g)[0]

        if self.record_diagnostics:
            self._record(t, sigma_t, b)
        return DiracPosterior(mean=x_hat + correction)

    def _record(self, t, sigma_t, b) -> None:
        # Misfit in units of the observation error: about 1 once fitted.
        z = b / self.cov_y.sqrt()
        entry = {
            "t": float(t),
            "sigma": float(sigma_t),
            "b_norm": float(torch.linalg.vector_norm(b)),
            "misfit_rms": float(z.square().mean().sqrt()),
        }
        if self.obs_groups is not None:
            names = self.group_names or [
                str(k) for k in range(int(self.obs_groups.max()) + 1)
            ]
            for k, name in enumerate(names):
                sel = self.obs_groups == k
                entry[f"misfit_rms_{name}"] = (
                    float(z[..., sel].square().mean().sqrt())
                    if bool(sel.any())
                    else float("nan")
                )
        self.diagnostics.append(entry)

# SPDX-License-Identifier: Apache-2.0
r"""Moment Matching Posterior Sampling (MMPS) guidance.

Derived from ``azula.guidance.mmps.MMPSDenoiser`` (azula 0.10.2, MIT,
Copyright (c) 2024 The Probabilists; https://github.com/probabilists/azula),
which implements

    Rozet et al. (2024), "Learning Diffusion Priors from Observations by
    Expectation Maximization", https://arxiv.org/abs/2405.13712.

At each reverse step, with :math:`\hat x = D_\theta(x_t)` and
:math:`\gamma_t = \sigma_t^2 / \alpha_t`, the guided mean is

.. math::
    \hat x + \gamma_t J^\top A^\top g, \qquad
    \big(\Sigma_y + A \, \gamma_t J \, A^\top\big) g = y - A(\hat x),

with :math:`J = \partial \hat x / \partial x_t`. For the *exact* conditional
mean, :math:`\gamma_t J = \mathrm{Cov}[x \mid x_t]` (an identity following
from Tweedie's identity), which is symmetric positive semidefinite. Here
three approximations stack on top of that identity: the learned denoiser
only approximates the conditional mean (so its Jacobian need not be
symmetric or PSD); MMPS treats :math:`p(x \mid x_t)` as Gaussian with those
moments (moment matching), which is an approximation for non-Gaussian
priors even with linear observations; and the linear system is solved with
only a few Krylov iterations. GMRES is the default because it assumes
nothing about symmetry; it does not by itself make the inference accurate.

With the defaults (no warm start, no preconditioner, no inflation) this
class reproduces azula's ``MMPSDenoiser`` exactly. The options added here
come from the windowed-SDA work in cBottle-for-ocean-DA, reduced to a
single window:

* ``solver``: ``"cg"``, ``"gmres"`` or ``"minres"``.
* ``warm_start``: start each solve from the previous step's solution. The
  solution changes slowly between neighbouring noise levels, so a few
  iterations then refine a good guess instead of starting from zero.
* ``preconditioner="diagonal"``: symmetric Jacobi scaling by an estimate of
  the operator diagonal, ``Σ_y + c_t diag(A Aᵀ)`` with
  ``c_t = σ_t² σ_data² / (σ_t² + σ_data²)``, the Tweedie variance of a
  Gaussian prior with standard deviation ``σ_data``. Split (two-sided)
  scaling keeps the system symmetric, so it is valid for CG and MINRES too.
* ``cov_y_inflation`` (κ): use ``Σ_y + κ σ_t² I`` during sampling. Early
  on, ``A Cov Aᵀ`` is large and ``Σ_y`` small, so the system is badly
  conditioned and a truncated solve is poor; the extra term bounds the
  condition number and vanishes as σ_t → 0, recovering ``Σ_y``. It is a
  numerical choice that changes the effective likelihood while it acts.
* ``record_diagnostics``: per-step relative residuals, overall and per
  observation group (for example SSHA versus SST).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import Schedule
from obottle_starter.inference.sda.operators import as_covariance
from obottle_starter.inference.sda.solvers import get_solver

__all__ = ["MMPSDenoiser"]

PRECONDITIONERS = ("none", "diagonal")


class MMPSDenoiser(Denoiser):
    r"""MMPS guidance around an unconditional denoiser.

    Args:
        denoiser: The unconditional denoiser :math:`q_\phi(X \mid X_t)`.
        y: Observations, shape ``(N,)`` or ``(B, N)``.
        A: Differentiable forward operator, ``(B, *x) -> (B, N)``. It may be
            affine; only its Jacobian enters the linear system.
        cov_y: Observation noise variances (diagonal of :math:`\Sigma_y`),
            shape ``(N,)`` or broadcastable to ``y``, or an
            :class:`~obottle_starter.inference.sda.operators.ObservationCovariance`
            (block-diagonal, possibly correlated errors).
        solver: ``"cg"``, ``"gmres"`` or ``"minres"``.
        iterations: Krylov iterations per step. The default of 1 reproduces
            azula; the high-level API uses
            :data:`obottle_starter.inference.sda.defaults.DEFAULTS` (4).
        warm_start: Start each solve from the previous step's solution.
        preconditioner: ``"none"`` or ``"diagonal"``.
        cov_y_inflation: κ in :math:`\Sigma_y + \kappa \sigma_t^2 I`.
        sigma_data: Data standard deviation for the diagonal preconditioner.
        record_diagnostics: Record residuals in :attr:`diagnostics`. Costs one
            extra operator application (one network backward) per step, two
            with ``warm_start``.
        obs_groups: Optional ``(N,)`` integer group index of each observation,
            for per-group residuals.
        group_names: Names for the groups in ``obs_groups``.
        probes: Rademacher probes used to estimate ``diag(A Aᵀ)``.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov_y: Tensor,
        *,
        solver: str = "gmres",
        iterations: int = 1,
        warm_start: bool = False,
        preconditioner: str = "none",
        cov_y_inflation: float = 0.0,
        sigma_data: float = 1.0,
        record_diagnostics: bool = False,
        obs_groups: Tensor | None = None,
        group_names: Sequence[str] | None = None,
        probes: int = 16,
    ) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("MMPS needs at least one solver iteration")
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {PRECONDITIONERS}; "
                f"got {preconditioner!r}"
            )
        if cov_y_inflation < 0:
            raise ValueError("cov_y_inflation must be nonnegative")
        self.denoiser = denoiser
        self.y = y
        self.A = A
        self.cov_y = cov_y
        self.cov = as_covariance(cov_y)
        self.solver = solver
        self.solve = get_solver(solver)
        self.iterations = int(iterations)
        self.warm_start = bool(warm_start)
        self.preconditioner = preconditioner
        self.cov_y_inflation = float(cov_y_inflation)
        self.sigma_data = float(sigma_data)
        self.record_diagnostics = bool(record_diagnostics)
        self.obs_groups = obs_groups
        self.group_names = list(group_names) if group_names is not None else None
        self.probes = int(probes)
        self.diagnostics: list[dict] = []
        self._g_prev: Tensor | None = None
        self._row_norm_sq: Tensor | None = None

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def reset(self) -> None:
        """Forget the warm-start state and recorded diagnostics."""
        self._g_prev = None
        self.diagnostics = []

    def _a_row_norm_sq(self, like: Tensor) -> Tensor:
        """Estimate ``diag(A Aᵀ)`` once: ``E[(A z)²]`` for Rademacher ``z``.

        Exact for rows with one nonzero (plain masks). ``A`` is affine, so its
        Jacobian is taken at zero for a single-sample probe.
        """
        if self._row_norm_sq is None:
            total = None
            zero = torch.zeros_like(like[:1])
            for _ in range(self.probes):
                z = torch.randint_like(zero, 2) * 2 - 1
                sq = torch.func.jvp(self.A, (zero,), (z,))[1].square()[0]
                total = sq if total is None else total + sq
            self._row_norm_sq = total / self.probes
        return self._row_norm_sq

    @torch.no_grad()
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)
        gamma_t = sigma_t**2 / alpha_t

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, t, **kwargs).mean
            y_hat = self.A(x_hat)

        # Shapes: x-space (B, C, T, H*W), y-space (B, N). Products with the
        # network's Jacobian come from autograd, which computes
        # vector-Jacobian products: autograd.grad(out, inp, v) = Jᵀ v, where
        # J = ∂out/∂inp. It never forms J, and it does not compute J v.

        def A_lin(v: Tensor) -> Tensor:
            # A v (forward-mode; A is affine, so this is its linear part).
            return torch.func.jvp(self.A, (x_hat.detach(),), (v,))[1]

        def At(v: Tensor) -> Tensor:
            # Aᵀ v: y-space -> x-space.
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        def cov_x(v: Tensor) -> Tensor:
            # γ_t Jᵀ v with J = ∂x̂/∂x_t: one network backward pass. Used as
            # Cov[x | x_t] v, which assumes J symmetric (true for the exact
            # denoiser, approximate for a learned one).
            return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]

        cov = self.cov
        if self.cov_y_inflation:
            cov = cov.shifted(self.cov_y_inflation * sigma_t**2)

        def operator(v: Tensor) -> Tensor:
            # Σ_y v + A Cov[x | x_t] Aᵀ v (Σ_y may be block-diagonal).
            return cov.matvec(v) + A_lin(cov_x(At(v)))

        b = self.y - y_hat
        g_prev = self._g_prev
        if g_prev is not None and g_prev.shape != b.shape:
            g_prev = None

        if self.preconditioner == "diagonal":
            c_t = sigma_t**2 * self.sigma_data**2 / (sigma_t**2 + self.sigma_data**2)
            d = cov.diagonal() + c_t * self._a_row_norm_sq(x_hat)
            s = torch.rsqrt(d.clamp(min=torch.finfo(d.dtype).tiny))

            def scaled(u: Tensor) -> Tensor:
                return s * operator(s * u)

            u0 = None if g_prev is None else g_prev / s
            g = s * self.solve(A=scaled, b=s * b, x0=u0, iterations=self.iterations)
        else:
            g = self.solve(A=operator, b=b, x0=g_prev, iterations=self.iterations)

        if self.record_diagnostics:
            self._record(t, sigma_t, b, g_prev, g, operator)
        if self.warm_start:
            self._g_prev = g.detach()

        # γ_t Jᵀ Aᵀ g in one backward pass through A and the network.
        correction = gamma_t * torch.autograd.grad(y_hat, x_t, g)[0]
        return DiracPosterior(mean=x_hat + correction)

    def _record(self, t, sigma_t, b, g_prev, g, operator) -> None:
        def ratio(r: Tensor, sel=None) -> float:
            if sel is not None:
                r, ref = r[..., sel], b[..., sel]
            else:
                ref = b
            den = torch.linalg.vector_norm(ref)
            if den == 0:
                return float("nan")
            return float(torch.linalg.vector_norm(r) / den)

        r_post = b - operator(g)
        entry = {
            "t": float(t),
            "sigma": float(sigma_t),
            "b_norm": float(torch.linalg.vector_norm(b)),
            "r0_ratio": (
                float("nan") if g_prev is None else ratio(b - operator(g_prev))
            ),
            "rpost_ratio": ratio(r_post),
        }
        if self.obs_groups is not None:
            names = self.group_names or [
                str(k) for k in range(int(self.obs_groups.max()) + 1)
            ]
            for k, name in enumerate(names):
                sel = self.obs_groups == k
                entry[f"rpost_ratio_{name}"] = (
                    ratio(r_post, sel) if bool(sel.any()) else float("nan")
                )
        self.diagnostics.append(entry)

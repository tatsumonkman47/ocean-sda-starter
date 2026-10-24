# SPDX-License-Identifier: Apache-2.0
r"""Krylov solvers for the MMPS linear system.

``cg`` and ``gmres`` are azula's solvers, vendored verbatim in
``obottle_starter.inference.sda._azula.linalg``. ``minres`` is a batched, GPU-resident
MINRES for symmetric (possibly indefinite) systems, carried over from
``cBottle-for-ocean-DA/ocean-inference``.

The MMPS operator is

    M v = Σ_y v + A Cov[x | x_t] Aᵀ v,   Cov[x | x_t] ≈ γ_t ∂μ/∂x_t

(an identity for the exact denoiser; an approximation for a learned one).

It is symmetric only if the denoiser Jacobian is, which a trained network
satisfies approximately. GMRES makes no symmetry assumption, so it is the
default. CG minimizes the M-norm error and MINRES the 2-norm residual over the
same Krylov space; both assume symmetry and are worth comparing at low
iteration counts.

All three run a fixed number of iterations without convergence checks, so they
never synchronize with the host. They accept leading batch dimensions.

The MINRES recurrence is the symmetric Lanczos process with an on-the-fly
Givens QR factorization (Greenbaum, "Iterative Methods for Solving Linear
Systems", 1997, Algorithm 6.5).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from obottle_starter.inference.sda._azula.linalg import cg, gmres

__all__ = ["SOLVERS", "cg", "get_solver", "gmres", "minres"]


def minres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    iterations: int = 1,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Solve a symmetric linear system ``A x = b`` with MINRES iterations.

    Arguments mirror :func:`cg` and :func:`gmres`:

    Args:
        A: Linear operator ``v ↦ A v``.
        b: Right-hand side, shape ``(*, D)``.
        x0: Initial guess, or ``None`` for zero.
        iterations: Number of Lanczos/MINRES steps.
        dtype: Working precision (default ``torch.float64``).

    Returns:
        The iterate after ``iterations`` steps, shape ``(*, D)``.
    """
    if dtype is None:
        dtype = torch.float64
    eps = torch.finfo(dtype).smallest_normal

    if x0 is None:
        x = torch.zeros_like(b).to(dtype)
        r = b.to(dtype)
    else:
        x = x0.to(dtype)
        r = (b - A(x0)).to(dtype)

    # Lanczos basis.
    beta = torch.linalg.vector_norm(r, dim=-1).clamp(min=eps)
    v_prev = torch.zeros_like(r)
    v = r / beta[..., None]

    # QR state for the minimum-residual update; the basis is never stored.
    eta = beta.clone()
    s_prev = torch.zeros_like(beta)
    s = torch.zeros_like(beta)
    c_prev = torch.ones_like(beta)
    c = torch.ones_like(beta)

    # The two previous search directions.
    m_prev_prev = torch.zeros_like(v)
    m_prev = torch.zeros_like(v)

    for _ in range(iterations):
        # Lanczos step: p = A v − α v − β v_prev.
        p = A(v.to(b)).to(dtype)
        alpha = torch.einsum("...i,...i", v, p)
        p = p - alpha[..., None] * v - beta[..., None] * v_prev
        beta_next = torch.linalg.vector_norm(p, dim=-1).clamp(min=eps)

        # Apply the two previous Givens rotations to the new tridiagonal
        # column (β, α, β_next).
        eps_k = s_prev * beta
        delta_bar = c_prev * beta
        delta = c * delta_bar + s * alpha
        gamma_bar = -s * delta_bar + c * alpha

        # New rotation zeroes β_next.
        gamma = torch.sqrt(gamma_bar * gamma_bar + beta_next * beta_next).clamp(min=eps)
        c_new = gamma_bar / gamma
        s_new = beta_next / gamma

        # m_k = (v − δ_k m_{k−1} − ε_k m_{k−2}) / γ_k
        m = (v - delta[..., None] * m_prev - eps_k[..., None] * m_prev_prev) / gamma[
            ..., None
        ]
        x = x + (c_new * eta)[..., None] * m
        eta = -s_new * eta

        v_prev = v
        v = p / beta_next[..., None]
        beta = beta_next
        m_prev_prev = m_prev
        m_prev = m
        s_prev, c_prev = s, c
        s, c = s_new, c_new

    return x.to(b)


SOLVERS: dict[str, Callable[..., Tensor]] = {
    "cg": cg,
    "gmres": gmres,
    "minres": minres,
}


def get_solver(name: str) -> Callable[..., Tensor]:
    """Return the solver registered under ``name`` (``cg``, ``gmres``, ``minres``)."""
    try:
        return SOLVERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown MMPS solver {name!r}; expected one of {sorted(SOLVERS)}"
        ) from None

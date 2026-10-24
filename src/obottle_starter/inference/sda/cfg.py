# SPDX-License-Identifier: Apache-2.0
r"""Classifier-free guidance (CFG) for conditional models.

    Ho & Salimans (2022), "Classifier-Free Diffusion Guidance",
    https://arxiv.org/abs/2207.12598.

A conditional model trained with some samples' conditions replaced by the
null condition ("nothing observed") estimates both
:math:`\hat x_c = E[x \mid x_t, y]` and :math:`\hat x_u = E[x \mid x_t]`.
CFG extrapolates from the second towards the first:

.. math::
    \hat x_w = \hat x_u + w \, (\hat x_c - \hat x_u),

so ``w = 1`` is the plain conditional model (Suresh Babu et al.'s M_3) and
``w > 1`` strengthens the observations' influence (their M_4). Each step
costs two network evaluations when ``w != 1``. CFG only makes sense for a
model trained with ``null_probability > 0``.
"""

from __future__ import annotations

from torch import Tensor

from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import Schedule

__all__ = ["CFGDenoiser"]


class CFGDenoiser(Denoiser):
    """Combine a conditional and a null-condition denoiser.

    Args:
        conditional: The model with the observation condition.
        unconditional: The same model with the null condition.
        weight: Guidance weight ``w``.
    """

    def __init__(
        self, conditional: Denoiser, unconditional: Denoiser, weight: float = 1.0
    ) -> None:
        super().__init__()
        self.conditional = conditional
        self.unconditional = unconditional
        self.weight = float(weight)

    @property
    def schedule(self) -> Schedule:
        return self.conditional.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        x_c = self.conditional(x_t, t, **kwargs).mean
        if self.weight == 1.0:
            return DiracPosterior(mean=x_c)
        x_u = self.unconditional(x_t, t, **kwargs).mean
        return DiracPosterior(mean=x_u + self.weight * (x_c - x_u))

# SPDX-License-Identifier: MIT
# Copyright (c) 2024 The Probabilists. Vendored from azula 0.10.2, azula/denoise.py.
r"""Denoiser interface.

For a perturbation kernel :math:`p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X,
\sigma_t^2 I)`, a denoiser approximates the posterior :math:`p(X \mid X_t)`; for most
uses it is enough that its mean :math:`\mu_\phi(x_t) \approx \mathbb{E}[X \mid x_t]`.
"""

__all__ = [
    "Posterior",
    "DiracPosterior",
    "Denoiser",
]

import abc

import torch.nn as nn
from torch import Tensor

from .noise import Schedule


class Posterior(abc.ABC):
    r"""Abstract posterior :math:`q_\phi(X \mid x_t)`."""

    mean: Tensor


class DiracPosterior(Posterior):
    r"""Creates a Dirac delta posterior :math:`\delta(X - \mu)`.

    Arguments:
        mean: The mean :math:`\mu`, with shape :math:`(*)`.
    """

    mean: Tensor

    def __init__(self, mean: Tensor) -> None:
        self.mean = mean


class Denoiser(nn.Module):
    r"""Abstract denoiser module."""

    schedule: Schedule

    @abc.abstractmethod
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Posterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The posterior :math:`q_\phi(X \mid x_t)`.
        """

        pass

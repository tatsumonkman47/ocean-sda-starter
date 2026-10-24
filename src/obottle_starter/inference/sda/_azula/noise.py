# SPDX-License-Identifier: MIT
# Copyright (c) 2024 The Probabilists. Vendored from azula 0.10.2, azula/noise.py.
r"""Noise schedules.

A noise schedule is a mapping from a time :math:`t \in [0, 1]` to the signal scale
:math:`\alpha_t \in \mathbb{R}_+` and the noise scale :math:`\sigma_t \in \mathbb{R}_+`
in a perturbation kernel

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X, \sigma_t^2 I)

from a "clean" random variable :math:`X` to a "noisy" random variable :math:`X_t`.
"""

__all__ = [
    "Schedule",
    "VESchedule",
]

import abc
import math

import torch
from torch import Tensor


class Schedule(abc.ABC):
    r"""Abstract noise schedule."""

    @abc.abstractmethod
    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Arguments:
            t: The time :math:`t`, with shape :math:`(*)`.

        Returns:
            The signal and noise scales :math:`\alpha_t` and :math:`\sigma_t`, with
            shape :math:`(*)`.
        """

        pass


class VESchedule(Schedule):
    r"""Creates a variance exploding (VE) noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp \big( (1 - t) \log \sigma_\min + t \log \sigma_\max \big)

    References:
        | Generative Modeling by Estimating Gradients of the Data Distribution (Song et al., 2019)
        | https://arxiv.org/pdf/1907.05600

        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e3) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(
            (1 - t) * math.log(self.sigma_min) + t * math.log(self.sigma_max)
        )

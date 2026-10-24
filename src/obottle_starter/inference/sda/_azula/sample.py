# SPDX-License-Identifier: MIT
# Copyright (c) 2024 The Probabilists. Vendored from azula 0.10.2, azula/sample.py:
# only Sampler and DDIMSampler are kept, verbatim.
r"""Reverse diffusion samplers.

For a distribution :math:`p(X)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(X_t) = \int p(X_t \mid x) \, p(x) \, dx \, .

The goal of diffusion models is to generate samples from :math:`p(X_0)`. To this end,
reverse transition kernels :math:`q(X_s \mid X_t)` from :math:`t` to :math:`s < t` are
chosen. Then, starting from :math:`x_1 \sim p(X_1)`, :math:`T` transitions
:math:`x_{t_{i-1}} \sim q(X_{t_{i-1}} \mid x_{t_i})` are simulated from :math:`t_T = 1`
to :math:`t_0 = 0`. If the kernels are consistent with the marginals :math:`p(X_t)`,
that is if

.. math:: p(X_{t_{i-1}}) \approx
    \int q(X_{t_{i-1}} \mid x_{t_i}) \, p(x_{t_i}) \, dx_{t_i} \, ,

for all :math:`i = 1, \dots, T`, the tensors :math:`x_{t_i}` are distributed according
to :math:`p(X_{t_i})`, including the last one :math:`x_{t_0} = x_0`.
"""


__all__ = [
    "Sampler",
    "DDIMSampler",
]

import abc

import torch

from collections.abc import Iterable, Sequence
from torch import Tensor
from tqdm import tqdm

from .denoise import Denoiser


class Sampler(abc.ABC):
    r"""Abstract reverse diffusion sampler.

    Arguments:
        start: The starting time :math:`t_T`.
        stop: The stopping time :math:`t_0`.
        steps: The number of discretization steps :math:`T`. By default, the step size
            :math:`t_i - t_{i-1}` is constant.
        silent: Whether to hide the sampling progress bar or not.
        dtype: The time data type.
        device: The time device.
    """

    denoiser: Denoiser

    def __init__(
        self,
        start: float = 1.0,
        stop: float = 0.0,
        steps: int = 64,
        silent: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.start = start
        self.stop = stop
        self.steps = steps
        self.silent = silent

        self.dtype = dtype
        self.device = device

    @property
    def timesteps(self) -> Tensor:
        return torch.linspace(
            self.start,
            self.stop,
            self.steps + 1,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.no_grad()
    def init(
        self,
        shape: Sequence[int],
        mean: float | Tensor = 0.0,
        var: float | Tensor = 1.0,
        **kwargs,
    ) -> Tensor:
        r"""Draws an initial noisy tensor :math:`x_{t_T}`.

        .. math:: x_{t_T} \sim \mathcal{N}(\alpha_{t_T} \mathbb{E}[X], \alpha_{t_T}^2 \mathbb{V}[X] + \sigma_{t_T}^2 I)

        Arguments:
            shape: The shape :math:`(*)` of the tensor.
            mean: The mean :math:`\mathbb{E}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`.
            var: The variance :math:`\mathbb{V}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`.
            kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

        Returns:
            A noisy tensor :math:`x_{t_T}`, with shape :math:`(*)`.
        """

        t_T = self.timesteps[0]

        alpha_T, sigma_T = self.denoiser.schedule(t_T)
        alpha_T, sigma_T = alpha_T.to(**kwargs), sigma_T.to(**kwargs)

        mean_T, std_T = alpha_T * mean, torch.sqrt(alpha_T**2 * var + sigma_T**2)
        mean_T, std_T = mean_T.expand(shape), std_T.expand(shape)

        return mean_T + std_T * torch.randn_like(mean_T)

    def progress_bar(self, it: Iterable) -> Iterable:
        if torch.is_tensor(it):
            it = it.unbind()

        if self.silent:
            return it
        else:
            return tqdm(it, miniters=1, unit="step", ncols=79, ascii=True)

    @torch.no_grad()
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t_T` to :math:`t_0`.

        Arguments:
            x: A noisy tensor :math:`x_{t_T}`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The clean(er) tensor :math:`x_{t_0}`, with shape :math:`(*)`.
        """

        time_pairs = self.timesteps.unfold(0, 2, 1).to(device=x.device)

        x_t = x

        for t, s in self.progress_bar(time_pairs):
            x_s = self.step(x_t, t, s, **kwargs)
            x_t = x_s

        x = x_t

        return x

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t` to :math:`s`.

        Arguments:
            x_t: The current tensor :math:`x_t`, with shape :math:`(*)`.
            t: The current time :math:`t`, with shape :math:`()`.
            s: The target time :math:`s`, with shape :math:`()`.
            kwargs: Optional keyword arguments.

        Returns:
            The new tensor :math:`x_s \sim q(X_s \mid x_t)`, with shape :math:`(*)`.
        """

        raise NotImplementedError()



class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: x_s \gets \alpha_s \mathbb{E}[X \mid x_t]
        + \sigma_s \, \sqrt{1 - \eta \, \tau} \, \frac{x_t - \alpha_t \mathbb{E}[X \mid x_t]}{\sigma_t}
        + \sigma_s \, \sqrt{\eta \, \tau} \, \varepsilon

    where :math:`\varepsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        eta: The stochasticity parameter :math:`\eta \in \mathbb{R}_+`.
            If :math:`\eta = 1`, :class:`DDIMSampler` is equivalent to :class:`DDPMSampler`.
            If :math:`\eta = 0`, :class:`DDIMSampler` is equivalent to :class:`EulerSampler`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, eta: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.eta = eta

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)

        q_t = self.denoiser(x_t, t, **kwargs)

        x_s = alpha_s * q_t.mean
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * q_t.mean)
        x_s = x_s + sigma_s * torch.sqrt(tau) * torch.randn_like(x_t)

        return x_s


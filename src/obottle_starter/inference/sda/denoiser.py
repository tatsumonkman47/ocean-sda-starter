# SPDX-License-Identifier: Apache-2.0
"""Bridge from a trained ocean network to the azula ``Denoiser`` interface.

The network (``EDMPrecond``) returns an object whose ``.out`` is the
denoised estimate ``D_θ(x_t, σ)``. The bridge freezes the per-window
conditioning once at construction and broadcasts it to the current batch
inside ``forward``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from obottle_starter.inference.sda._azula.denoise import Denoiser, DiracPosterior
from obottle_starter.inference.sda._azula.noise import Schedule


class EDMPrecondDenoiser(Denoiser):
    """Wrap a preconditioned ocean network as an azula ``Denoiser``.

    The forward pass is differentiable end to end through
    ``c_skip * x + c_out * model(...)``, so guidance such as
    :class:`obottle_starter.inference.sda.mmps.MMPSDenoiser`, which differentiates ``x_hat``
    with respect to ``x_t``, applies unchanged.

    Args:
        net: A network returned by
            :func:`obottle_starter.core.model.load_checkpoint`. Must have
            ``use_fp16 == False`` for clean MMPS gradients.
        schedule: Noise schedule used by downstream samplers, within the
            training noise range (``cfg.sigma_min`` to ``cfg.sigma_max``).
        condition: Per-tile spatial conditioning, shape ``(C_cond, T, H*W)``,
            or ``None`` if the trained model has no spatial conditioning.
        class_labels: Per-tile label vector, shape ``(label_dim,)``, or
            ``None``. ``EDMPrecond`` substitutes zeros when ``label_dim > 0``
            and ``None`` is passed, which is the normal path for
            ``label_dim == 0`` checkpoints.
        second_of_day: ``(T,)`` local-solar seconds, or ``None``.
        day_of_year: ``(T,)`` day-of-year index, or ``None``.
        time_offsets: ``(T,)`` offset-from-frame-0 in hours, or ``None``.

    Notes:
        The caller is responsible for placing every conditioning tensor on
        the network's device before constructing the bridge — ``.to(device)``
        on the denoiser does not move plain-attribute tensors.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        schedule: Schedule,
        *,
        condition: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        second_of_day: Optional[Tensor] = None,
        day_of_year: Optional[Tensor] = None,
        time_offsets: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.net = net
        # Schedule is a plain object (not nn.Module); mirror GaussianDenoiser.
        self.schedule = schedule
        self._condition = condition
        self._class_labels = class_labels
        self._second_of_day = second_of_day
        self._day_of_year = day_of_year
        self._time_offsets = time_offsets

    def _broadcast_cond(self, B: int) -> dict:
        """Expand each conditioning tensor along a leading batch dim of size ``B``."""

        def _bcast(x: Optional[Tensor]) -> Optional[Tensor]:
            if x is None:
                return None
            return x.unsqueeze(0).expand(B, *x.shape)

        return dict(
            condition=_bcast(self._condition),
            class_labels=_bcast(self._class_labels),
            second_of_day=_bcast(self._second_of_day),
            day_of_year=_bcast(self._day_of_year),
            time_offsets=_bcast(self._time_offsets),
        )

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        _, sigma_t = self.schedule(t)
        B = x_t.shape[0]
        # EDMPrecond.forward reshapes sigma to (-1, 1, 1, 1). Pass a 1-D (B,)
        # tensor so the broadcast is explicit at the call site.
        sigma = sigma_t.reshape(-1).expand(B).to(x_t.device).to(x_t.dtype)
        cond = self._broadcast_cond(B)
        # Drop None-valued kwargs so EDMPrecond / SongUNet see the same
        # signature shape as at training time.
        cond = {k: v for k, v in cond.items() if v is not None}
        out = self.net(x_t, sigma, force_fp32=True, **cond)
        return DiracPosterior(mean=out.out)

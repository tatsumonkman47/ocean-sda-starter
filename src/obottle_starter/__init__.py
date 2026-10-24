# SPDX-License-Identifier: Apache-2.0
"""Ocean diffusion priors (SongUNet) and score-based data assimilation.

The package has three parts:

* :mod:`obottle_starter.core`: shared by both sides. The network, checkpoint
  loading, normalization and time conditioning.
* :mod:`obottle_starter.training`: datasets and the training loop.
* :mod:`obottle_starter.inference`: SDA. ``assimilate``, the OSSE tiles,
  metrics, and the MMPS machinery in ``inference.sda``.

The most used functions are re-exported here, so
``import obottle_starter as obs; obs.load_checkpoint(...); obs.assimilate(...)``
works without knowing the layout.
"""

from obottle_starter.core.conditioning import frame_conditioning
from obottle_starter.core.model import (
    build_model,
    channel_names,
    export_checkpoint,
    load_checkpoint,
)
from obottle_starter.core.normalization import Normalizer
from obottle_starter.inference.assimilate import assimilate

__all__ = [
    "Normalizer",
    "assimilate",
    "build_model",
    "channel_names",
    "export_checkpoint",
    "frame_conditioning",
    "load_checkpoint",
]

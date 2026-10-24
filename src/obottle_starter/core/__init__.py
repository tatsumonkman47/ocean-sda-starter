# SPDX-License-Identifier: Apache-2.0
"""What training and inference share: the network and how it sees data.

* ``nn/``: the EDMPrecond(SongUNet) backbone (vendored NVIDIA cBottle).
* ``model``: build a network from a config; load and export checkpoints.
* ``normalization``: physical units <-> model units, per channel.
* ``conditioning``: frame times -> the network's time inputs.

Nothing here imports from ``training`` or ``inference``.
"""

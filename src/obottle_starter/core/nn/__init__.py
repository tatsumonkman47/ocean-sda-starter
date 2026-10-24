# SPDX-License-Identifier: Apache-2.0
"""The EDM / SongUNet backbone, vendored from NVIDIA cBottle.

``networks``, ``embedding``, ``domain`` and ``loss`` are a trimmed copy of
cBottle (Apache-2.0, Copyright NVIDIA) as adapted for ocean patches in
cBottle-for-ocean-DA: HEALPix support removed, per-frame adaptive group
norm and continuous-offset temporal attention added. They load the released
checkpoints unchanged.
"""

# Third-party code

* `src/obottle_starter/core/nn/`: derived from NVIDIA cBottle
  (https://github.com/NVlabs/cBottle), Copyright (c) 2025 NVIDIA CORPORATION &
  AFFILIATES, Apache License 2.0 (`LICENSE.txt`). The files keep their
  original notices. Changes: HEALPix support removed; per-frame adaptive group
  norm, continuous-offset temporal attention and calendar embedding for
  ocean patches. The EDM network design follows Karras et al. (2022),
  "Elucidating the Design Space of Diffusion-Based Generative Models".
* `src/obottle_starter/inference/sda/_azula/`: a subset of azula 0.10.2
  (https://github.com/probabilists/azula), Copyright (c) 2024 The
  Probabilists, MIT License (`src/obottle_starter/inference/sda/_azula/LICENSE`).
  `src/obottle_starter/inference/sda/mmps.py` is derived from azula's `MMPSDenoiser`.

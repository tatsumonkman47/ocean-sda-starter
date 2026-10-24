# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for priors and posterior ensembles, beyond training loss.

All functions take plain arrays (or xarray objects) in physical units and
return numbers or arrays; :func:`summarize` prints the usual set for an
:func:`obottle_starter.assimilate` result. What each group answers:

* **Reconstruction** (:func:`masked_rmse`, :func:`reconstruction_errors`):
  errors at observed and unobserved pixels separately. Fitting observed
  pixels is easy; the unobserved error is what SDA is for.
* **Observation fit** (:func:`observation_misfit`): residuals at observed
  pixels in units of ``sigma_y``. A useful diagnostic, but its expected
  value depends on the posterior uncertainty and the prior, not only on
  ``sigma_y``, so it is not a target to hit exactly.
* **Ensemble** (:func:`spread_skill`, :func:`rank_histogram`,
  :func:`interval_coverage`): whether the spread matches the actual error.
  Spread is the variability under the learned model and chosen sampler;
  only these comparisons, over many cases, say whether it is calibrated.
* **Prior and joint statistics** (:func:`isotropic_psd`,
  :func:`isotropic_coherence`, :func:`spatial_correlation`,
  :func:`frame_correlation`): spectra, cross-variable relationships and
  temporal decorrelation, to compare samples with truth. For a joint model,
  check relationships between variables, not only each one's spectrum.

One window is one case: draw conclusions from many windows.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def rmse(a, b, axis=(-2, -1)) -> np.ndarray:
    """Root-mean-square difference over ``axis`` (default: the grid)."""
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def masked_rmse(a, b, mask) -> float:
    """RMS of ``a - b`` over the pixels where ``mask`` is true (NaN if none)."""
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), np.shape(a))
    if not mask.any():
        return float("nan")
    diff = (np.asarray(a) - np.asarray(b))[mask]
    return float(np.sqrt(np.mean(diff**2)))


def reconstruction_errors(samples, truth, mask) -> dict[str, float]:
    """Ensemble-mean RMSE at observed pixels, unobserved pixels and overall.

    Args:
        samples: ``(M, T, H, W)`` members of one variable.
        truth: ``(T, H, W)``.
        mask: ``(T, H, W)`` observed pixels.
    """
    mean = np.asarray(samples).mean(axis=0)
    mask = np.asarray(mask, dtype=bool)
    return {
        "observed": masked_rmse(mean, truth, mask),
        "unobserved": masked_rmse(mean, truth, ~mask),
        "all": masked_rmse(mean, truth, np.ones_like(mask)),
    }


def observation_misfit(samples, observations, sigma_y: float) -> float:
    """RMS over members and observed pixels of ``(member - y) / sigma_y``.

    ``observations`` is NaN where unobserved (as in ``assimilate``'s output).
    """
    y = np.asarray(observations)
    observed = np.isfinite(y)
    if not observed.any():
        return float("nan")
    z = (np.asarray(samples)[:, observed] - y[observed]) / float(sigma_y)
    return float(np.sqrt(np.mean(z**2)))


def spread_skill(samples, truth, mask=None) -> dict[str, float]:
    """Ensemble spread against ensemble-mean error.

    ``spread`` is ``sqrt(mean member variance)`` and ``rmse`` the
    ensemble-mean RMSE, over ``mask`` (default: everywhere). For a calibrated
    ensemble of ``M`` members, ``rmse ≈ spread * sqrt((M + 1) / M)``;
    ``ratio`` is ``rmse`` divided by that, so about 1 is consistent,
    above 1 under-dispersed and below 1 over-dispersed.
    """
    samples = np.asarray(samples)
    M = samples.shape[0]
    mean = samples.mean(axis=0)
    var = samples.var(axis=0, ddof=1) if M > 1 else np.zeros_like(mean)
    sel = (
        np.ones(mean.shape, dtype=bool)
        if mask is None
        else np.broadcast_to(np.asarray(mask, dtype=bool), mean.shape)
    )
    if not sel.any():
        return {"rmse": float("nan"), "spread": float("nan"), "ratio": float("nan")}
    err = float(np.sqrt(np.mean((mean - np.asarray(truth))[sel] ** 2)))
    spread = float(np.sqrt(np.mean(var[sel])))
    expected = spread * np.sqrt((M + 1) / M)
    return {
        "rmse": err,
        "spread": spread,
        "ratio": err / expected if expected else float("inf"),
    }


def rank_histogram(samples, truth, mask=None) -> np.ndarray:
    """Counts of the truth's rank among ``M`` members, length ``M + 1``.

    Flat for a calibrated ensemble; U-shaped when under-dispersed (truth
    often outside the members), domed when over-dispersed. Ties are broken
    at random. Pixels are correlated, so judge the shape over many windows.
    """
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    sel = (
        np.ones(truth.shape, dtype=bool)
        if mask is None
        else np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
    )
    members = samples[:, sel]
    t = truth[sel]
    below = (members < t).sum(axis=0)
    ties = (members == t).sum(axis=0)
    rng = np.random.default_rng(0)
    ranks = below + rng.integers(0, ties + 1)
    return np.bincount(ranks, minlength=samples.shape[0] + 1)


def interval_coverage(samples, truth, level: float = 0.8, mask=None) -> float:
    """Fraction of pixels whose truth lies in the members' central interval.

    The interval comes from the members' empirical quantiles, so even a
    calibrated ensemble of ``M`` members covers somewhat less than
    ``level``, about ``level * (M - 1) / (M + 1)``.
    """
    samples = np.asarray(samples)
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    truth = np.asarray(truth)
    inside = (truth >= lo) & (truth <= hi)
    if mask is not None:
        inside = inside[np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)]
    return float(np.mean(inside))


def spatial_correlation(a, b) -> np.ndarray:
    """Pearson correlation between two fields over the grid, per leading index.

    ``a`` and ``b`` are ``(..., H, W)``; returns ``(...)``. For a joint
    model, compare the correlation between variables in samples and truth.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    num = (a * b).sum(axis=(-2, -1))
    den = np.sqrt((a**2).sum(axis=(-2, -1)) * (b**2).sum(axis=(-2, -1)))
    return num / np.where(den > 0, den, np.nan)


def frame_correlation(field) -> np.ndarray:
    """Correlation of each frame with frame 0 over the grid.

    ``field`` is ``(..., T, H, W)``; returns ``(..., T)``. Plot against the
    frame offsets to compare temporal decorrelation of samples and truth.
    """
    field = np.asarray(field)
    return spatial_correlation(field, field[..., :1, :, :])


def _radial_bins(H: int, W: int, dx_km: float):
    ky = np.fft.fftfreq(H, d=dx_km)
    kx = np.fft.fftfreq(W, d=dx_km)
    kr = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    dk = 1.0 / (dx_km * max(H, W))
    edges = np.arange(0.0, kr.max() + dk, dk)
    which = np.digitize(kr.ravel(), edges) - 1
    count = np.bincount(which, minlength=len(edges))
    k = edges[:-1] + dk / 2
    keep = (count[:-1] > 0) & (k > 0) & (k <= 0.5 / dx_km)
    return which, count, k, keep, len(edges)


def _spectra(field, dx_km: float):
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape[-2:]
    field = field.reshape(-1, H, W)
    field = field - field.mean(axis=(-2, -1), keepdims=True)
    window = np.outer(np.hanning(H), np.hanning(W))
    return np.fft.fft2(field * window), (window**2).sum()


def isotropic_psd(field, dx_km: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density of ``(..., H, W)`` fields.

    Each field is demeaned and tapered with a 2-D Hann window. Leading axes
    are averaged.

    Returns:
        ``(k, psd)``: wavenumber in cycles/km at the ring centres, and power
        density per unit wavenumber, averaged over leading axes.
    """
    coeffs, norm = _spectra(field, dx_km)
    H, W = coeffs.shape[-2:]
    spec = (np.abs(coeffs) ** 2 * dx_km**2 / norm).mean(axis=0)
    which, count, k, keep, n = _radial_bins(H, W, dx_km)
    total = np.bincount(which, spec.ravel(), minlength=n)
    # Sum over a ring of width dk: mean power times circumference / dk.
    psd = total[:-1] / np.maximum(count[:-1], 1) * 2 * np.pi * k
    return k[keep], psd[keep]


def isotropic_coherence(a, b, dx_km: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Squared coherence between two sets of fields, by wavenumber ring.

    ``|<A B*>|² / (<|A|²> <|B|²>)`` with averages over leading axes and each
    ring: 1 where the two vary together at that scale, near 0 where they are
    unrelated. Needs several fields (leading axes) to be meaningful.

    Returns:
        ``(k, coherence²)``, ``k`` in cycles/km.
    """
    A, _ = _spectra(a, dx_km)
    B, _ = _spectra(b, dx_km)
    H, W = A.shape[-2:]
    which, _, k, keep, n = _radial_bins(H, W, dx_km)

    def ring(x):
        x = x.mean(axis=0).ravel()
        re = np.bincount(which, x.real, minlength=n)
        im = np.bincount(which, x.imag, minlength=n) if np.iscomplexobj(x) else 0
        return (re + 1j * im)[:-1]

    cross = ring(A * np.conj(B))
    pa, pb = ring(np.abs(A) ** 2).real, ring(np.abs(B) ** 2).real
    coh = np.abs(cross) ** 2 / np.maximum(pa * pb, np.finfo(float).tiny)
    return k[keep], coh[keep]


def summarize(ds, sigma_y: Mapping[str, float] | None = None) -> str:
    """A text report on an :func:`obottle_starter.assimilate` result.

    Per variable: errors at observed and unobserved pixels, spread against
    error, 80% interval coverage and, with ``sigma_y``, the observation
    misfit. Needs ``truth`` (an OSSE). One window is one case.
    """
    lines = ["diagnostics (one window; judge over many):"]
    for c in [str(v) for v in ds.channel.values]:
        samples = ds.samples.sel(channel=c).values
        truth = ds.truth.sel(channel=c).values
        mask = ds.mask.sel(channel=c).values.astype(bool)
        units = dict(
            p.split("=", 1) for p in ds.attrs.get("units", "").split(",") if "=" in p
        ).get(c, "")
        err = reconstruction_errors(samples, truth, mask)
        ss = spread_skill(samples, truth, ~mask if mask.any() else None)
        cov = interval_coverage(samples, truth, 0.8, ~mask if mask.any() else None)
        line = (
            f"  {c}: ensemble-mean RMSE {err['all']:.4g} {units} "
            f"(observed {err['observed']:.4g}, unobserved {err['unobserved']:.4g})"
            f"\n       unobserved: spread {ss['spread']:.4g}, "
            f"error/spread {ss['ratio']:.2f} (about 1 if calibrated), "
            f"80% interval covers {100 * cov:.0f}%"
        )
        if sigma_y and c in sigma_y and "observations" in ds and mask.any():
            mis = observation_misfit(
                samples, ds.observations.sel(channel=c).values, sigma_y[c]
            )
            line += f"\n       members minus observations: {mis:.2f} sigma_y (RMS)"
        lines.append(line)
    return "\n".join(lines)

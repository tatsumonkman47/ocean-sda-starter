# SPDX-License-Identifier: Apache-2.0
"""Diagnostics on cases with known answers."""

import numpy as np

from obottle_starter import assimilate
from obottle_starter.inference.metrics import (
    frame_correlation,
    interval_coverage,
    isotropic_coherence,
    observation_misfit,
    rank_histogram,
    reconstruction_errors,
    spatial_correlation,
    spread_skill,
    summarize,
)
from obottle_starter.inference.osse import random_masks
from obottle_starter.training.data import structured_fields

TIMES = np.datetime64("2012-03-01T00") + np.arange(3) * np.timedelta64(12, "h")


def test_calibrated_ensemble_looks_calibrated():
    # Truth and members drawn from the same distribution: calibrated.
    rng = np.random.default_rng(0)
    M = 20
    draws = rng.standard_normal((M + 1, 4, 32, 32))
    truth, samples = draws[0], draws[1:]
    ss = spread_skill(samples, truth)
    assert abs(ss["ratio"] - 1) < 0.05
    hist = rank_histogram(samples, truth)
    assert len(hist) == M + 1 and hist.sum() == truth.size
    assert hist.max() / hist.mean() < 1.3  # roughly flat
    expected = 0.8 * (M - 1) / (M + 1)
    assert abs(interval_coverage(samples, truth, 0.8) - expected) < 0.03
    # An ensemble with half the spread is under-dispersed.
    narrow = spread_skill(0.5 * samples, truth)
    assert narrow["ratio"] > 1.5
    assert rank_histogram(0.5 * samples, truth)[[0, -1]].sum() > 3 * hist[[0, -1]].sum()


def test_reconstruction_errors_split_by_mask():
    truth = np.zeros((2, 4, 4))
    samples = np.ones((3, 2, 4, 4))
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[:, 0] = True
    samples[:, mask] = 0.0  # perfect where observed
    err = reconstruction_errors(samples, truth, mask)
    assert err["observed"] == 0 and err["unobserved"] == 1
    assert np.isclose(err["all"], np.sqrt(0.75))


def test_observation_misfit_in_sigma_units():
    y = np.full((2, 4, 4), np.nan)
    y[:, 0] = 0.0
    samples = np.full((3, 2, 4, 4), 0.02)
    assert np.isclose(observation_misfit(samples, y, sigma_y=0.01), 2.0)


def test_cross_variable_diagnostics_recover_known_correlation():
    rng = np.random.default_rng(1)
    fields = np.stack(
        [
            structured_fields((32, 32), [0, 24], 2, rng, correlation=0.8)
            for _ in range(100)
        ]
    )
    corr = spatial_correlation(fields[:, 0], fields[:, 1])
    assert abs(np.mean(corr) - 0.8) < 0.05
    _, coh = isotropic_coherence(fields[:, 0, 0], fields[:, 1, 0], dx_km=1.0)
    assert abs(np.median(coh) - 0.64) < 0.08  # coherence² = 0.8² at every scale
    lag = frame_correlation(fields[:, 0])
    assert np.allclose(lag[:, 0], 1) and lag[:, 1].mean() < 1


def test_assimilate_output_supports_summary(tiny_net, tiny_cfg):
    rng = np.random.default_rng(0)
    truth = {
        "SSH": rng.normal(0, 0.05, (3, 16, 16)),
        "SST": rng.normal(18, 1, (3, 16, 16)),
    }
    masks = random_masks(["SSH"], (3, 16, 16), 0.3, seed=0)
    ds = assimilate(
        tiny_net,
        tiny_cfg,
        times=TIMES,
        center_lon=0.0,
        masks=masks,
        truth=truth,
        sigma_y={"SSH": 0.005},
        instrument_noise=False,
        n_members=2,
        steps=3,
        verbose=False,
    )
    obs = ds.observations.sel(channel="SSH").values
    np.testing.assert_allclose(obs[masks["SSH"]], truth["SSH"][masks["SSH"]], atol=1e-5)
    assert np.isnan(obs[~masks["SSH"]]).all()
    assert np.isnan(ds.observations.sel(channel="SST")).all()
    assert ds.attrs["route"] == "prior + SDA (mmps)"
    text = summarize(ds, {"SSH": 0.005})
    assert "SSH: ensemble-mean RMSE" in text and "sigma_y (RMS)" in text

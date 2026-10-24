# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from obottle_starter.core.config import upgrade, variables
from obottle_starter.core.normalization import (
    Climatology,
    Normalizer,
    default_climatology_path,
    file_sha256,
)
from obottle_starter.inference.sda.observations import ObservationOperator

# The flat format the released checkpoints store.
LEGACY = OmegaConf.create(
    {
        "channels": ["SSH", "SST"],
        "num_channels": 2,
        "out_transform_list": ["std_ssh_norm", "std_seasonal_mean_sst_norm"],
        "llc_standards": {"std_ssh": 0.051, "std_sst": 5.5, "extra_mean_tuning": 0.0},
    }
)
CFG = upgrade(LEGACY)
TIMES = np.datetime64("2011-10-01T06") + np.arange(3) * np.timedelta64(12, "h")


def test_legacy_translation():
    specs = variables(LEGACY)
    assert [s["name"] for s in specs] == ["SSH", "SST"]
    assert specs[0]["units"] == "m" and specs[0]["normalization"]["scale"] == 0.051
    clim = specs[1]["normalization"]["climatology"]
    assert clim["lookup"] == "elapsed_days" and clim["out_of_range"] == "error"
    assert "out_transform_list" not in CFG and CFG.config_version == 2
    assert variables(CFG) == specs


def test_roundtrip_and_constants():
    norm = Normalizer.from_config(CFG)
    sst = 20 + np.random.default_rng(0).standard_normal((3, 4, 4)).astype(np.float32)
    back = norm.denormalize("SST", norm.normalize("SST", sst, TIMES), TIMES)
    np.testing.assert_allclose(back, sst, rtol=1e-6)
    assert norm.scale("SSH") == 0.051 and norm.units("SST") == "degC"
    np.testing.assert_allclose(
        norm.normalize("SSH", np.full((3, 2), 0.102), TIMES), 2.0
    )
    assert norm.sigma_to_model("SST", 0.11) == pytest.approx(0.02)


def test_affine_variable_needs_no_code():
    spec = {"name": "U", "units": "m/s", "normalization": {"scale": 0.2, "offset": 0.1}}
    norm = Normalizer([spec])
    np.testing.assert_allclose(norm.normalize("U", np.full((3, 2), 0.5), TIMES), 2.0)
    assert norm.formula("U") == "x_model = (x - 0.1) / 0.2 m/s"


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"name": "U", "normalization": {"scale": 0.0}}, "scale"),
        (
            {"name": "U", "normalization": {"scale": 1.0, "offset": float("nan")}},
            "offset",
        ),
        (
            {
                "name": "U",
                "normalization": {"scale": 1.0, "climatology": {"lookup": "x"}},
            },
            "lookup",
        ),
        ({"name": "U", "normalisation": {"scale": 1.0}}, "unknown key"),
    ],
)
def test_bad_variable_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        Normalizer([spec])


def test_elapsed_days_matches_training_rule():
    # Training: daily index = (hours since 2011-09-13T00) // 24.
    clim = Climatology(lookup="elapsed_days")
    hours = np.array([0, 23, 24, 500, 9000])
    times = np.datetime64("2011-09-13T00", "ns") + hours.astype("timedelta64[h]")
    clim(times)
    np.testing.assert_array_equal(clim(times), clim._values[hours // 24])


def test_elapsed_days_rejects_dates_outside_the_record():
    clim = Climatology(lookup="elapsed_days")
    later = np.array(["2014-06-01T00"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="outside the climatology record"):
        clim(later)
    clipped = Climatology(lookup="elapsed_days", out_of_range="clip")
    np.testing.assert_array_equal(clipped(later), clipped._values[-1:])
    earlier = np.array(["2011-09-01T00"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(clipped(earlier), clipped._values[:1])


def test_day_of_year_lookup():
    doy = Climatology(lookup="day_of_year")
    ref = Climatology(lookup="elapsed_days")
    # A date that appears once in the 377-day record gives the same value.
    once = np.array(["2012-01-15T12"], dtype="datetime64[ns]")
    np.testing.assert_allclose(doy(once), ref(once))
    # Any year works: same calendar day, same value.
    other = np.array(["2030-01-15T00"], dtype="datetime64[ns]")
    np.testing.assert_allclose(doy(other), doy(once))
    # Year boundary.
    new_year = np.array(["2019-12-31T23", "2020-01-01T00"], dtype="datetime64[ns]")
    np.testing.assert_allclose(
        doy(new_year),
        ref(np.array(["2011-12-31", "2012-01-01"], dtype="datetime64[ns]")),
    )
    # Days present twice (Sep 13-23 in 2011 and 2012) use the mean.
    twice = ref(np.array(["2011-09-20", "2012-09-20"], dtype="datetime64[ns]"))
    sep20 = np.array(["2025-09-20"], dtype="datetime64[ns]")
    np.testing.assert_allclose(doy(sep20), twice.mean(), rtol=1e-6)
    # The record holds 2012-02-29, so a leap day elsewhere uses it.
    leap = np.array(["2016-02-29"], dtype="datetime64[ns]")
    np.testing.assert_allclose(
        doy(leap), ref(np.array(["2012-02-29"], dtype="datetime64[ns]"))
    )


def test_climatology_checksum_is_checked(tmp_path):
    good = file_sha256(default_climatology_path())
    Climatology(sha256=good)(TIMES)
    with pytest.raises(ValueError, match="sha256"):
        Climatology(sha256="0" * 64)(TIMES)


def test_observation_terms_are_in_model_units():
    norm = Normalizer.from_config(CFG)
    rng = np.random.default_rng(1)
    truth = {"SSH": rng.normal(0, 0.1, (3, 4, 4)), "SST": rng.normal(20, 2, (3, 4, 4))}
    x_model = {c: norm.normalize(c, truth[c], TIMES) for c in ("SSH", "SST")}
    x = torch.from_numpy(np.stack([x_model["SSH"], x_model["SST"]])).reshape(2, 3, 16)
    masks = {c: rng.random((3, 4, 4)) < 0.5 for c in ("SST", "SSH")}
    op = ObservationOperator(norm.observation_terms(masks))
    assert op.names == ["SSH", "SST"]  # model channel order
    expected = np.concatenate([x_model[c][masks[c]] for c in ("SSH", "SST")])
    np.testing.assert_allclose(op(x).numpy(), expected, rtol=1e-6)


def test_settings_default_to_checkpoint(capsys):
    norm = Normalizer.from_settings(CFG, {"source": "from_checkpoint"})
    assert norm.source == "checkpoint"
    assert norm.climatology("SST").path == default_climatology_path()
    assert Normalizer.from_settings(CFG, None).formula("SSH") == norm.formula("SSH")
    assert "WARNING" not in capsys.readouterr().out
    other = Normalizer.from_settings(
        CFG, {"climatology": {"path": "/elsewhere.nc", "lookup": None}}
    )
    assert other.climatology("SST").path == "/elsewhere.nc"
    assert "no checksum recorded" in capsys.readouterr().out
    doy = Normalizer.from_settings(CFG, {"climatology": {"lookup": "day_of_year"}})
    assert doy.climatology("SST").lookup == "day_of_year"
    assert "lookup: day_of_year (trained: elapsed_days)" in capsys.readouterr().out


def test_settings_override_reports_differences(capsys):
    specs = OmegaConf.to_container(CFG.variables)
    specs[0]["normalization"]["scale"] = 0.05
    settings = {"source": "config", "variables": specs}
    norm = Normalizer.from_settings(CFG, settings)
    assert norm.scale("SSH") == 0.05
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "x / 0.05 m  (trained: x_model = x / 0.051 m)" in out
    specs[0]["normalization"]["scale"] = 0.051
    Normalizer.from_settings(CFG, settings)
    assert "WARNING" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="source"):
        Normalizer.from_settings(CFG, {"source": "guess"})
    with pytest.raises(ValueError, match="must match"):
        Normalizer.from_settings(CFG, {"source": "config", "variables": specs[::-1]})


def test_describe_and_attrs():
    norm = Normalizer.from_config(CFG)
    text = norm.describe({"SSH": 0.0051})
    assert "SSH: x_model = x / 0.051 m" in text
    assert "SST: x_model = (x - clim(t)) / 5.5 degC" in text
    assert "sigma_y 0.0051 m -> 0.1 model units" in text
    assert "solved in model units" in text
    assert "elapsed_days" in text
    assert "solved in model units" not in norm.describe(sda=False)
    attrs = norm.as_attrs()
    assert '"lookup": "elapsed_days"' in attrs["normalization_variables"]


def test_default_climatology_is_in_repo_data():
    from pathlib import Path

    path = Path(default_climatology_path())
    assert path.exists() and path.parent.name == "data"
    assert path.parent.parent == Path(__file__).resolve().parents[1]

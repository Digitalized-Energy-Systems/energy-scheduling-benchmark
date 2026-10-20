"""Measured renewable profiles for lower_saxony_2030 (decision Q23, Phase 2c)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NETWORKS = Path(__file__).resolve().parents[1] / "resources" / "networks"
PROFILES = NETWORKS / "tennet_2019_capacity_factors.csv"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, NETWORKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_tennet_profiles")
apply = _load("apply_renewable_profiles")


def test_profiles_file_is_a_full_year_of_capacity_factors():
    cf = pd.read_csv(PROFILES, index_col=0, parse_dates=True)
    assert list(cf.columns) == ["solar", "onwind", "offwind"]
    assert len(cf) == 8760 and cf.index[0].year == 2019
    assert ((cf >= 0) & (cf <= 1)).all().all()
    # Measured 2019 TenneT feed-in: plausible annual capacity factors.
    assert 0.08 < cf["solar"].mean() < 0.12
    assert 0.2 < cf["onwind"].mean() < 0.32
    assert 0.3 < cf["offwind"].mean() < 0.5


def test_lower_saxony_uses_the_measured_profiles():
    import pypsa

    n = pypsa.Network(NETWORKS / "lower_saxony_2030.nc")
    cf = pd.read_csv(PROFILES, index_col=0, parse_dates=True)
    gens = n.generators
    for carrier, column in (
        ("solar", "solar"),
        ("onwind", "onwind"),
        ("offwind-ac", "offwind"),
    ):
        name = gens.index[gens.carrier == carrier][0]
        np.testing.assert_allclose(
            n.generators_t.p_max_pu[name].to_numpy(), cf[column].to_numpy(), atol=1e-9
        )


def test_build_arithmetic_on_synthetic_data():
    idx = pd.date_range("2019-01-01", periods=3, freq="h", tz="UTC")
    series = pd.DataFrame(
        {
            "DE_tennet_solar_generation_actual": [0.0, 500.0, 2000.0],
            "DE_tennet_wind_onshore_generation_actual": [100.0, 200.0, 300.0],
            "DE_tennet_wind_offshore_generation_actual": [5236.0, 0.0, 9999.0],
            "DE_solar_capacity": [4000.0, np.nan, 4000.0],
            "DE_wind_onshore_capacity": [1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    cf = build.build(series, {"solar": 0.5, "onwind": 1.0})
    np.testing.assert_allclose(cf["solar"], [0.0, 0.25, 1.0])  # 2000 / (4000·0.5) = 1
    np.testing.assert_allclose(cf["onwind"], [0.1, 0.2, 0.3])
    assert cf["offwind"].iloc[0] == pytest.approx(1.0)  # 5236 MW at the end of 2018
    assert cf["offwind"].iloc[2] == 1.0  # clipped


def test_tennet_shares():
    plants = pd.DataFrame(
        {
            "technology": ["Photovoltaics", "Photovoltaics", "Onshore", "Onshore"],
            "tso": ["TenneT", "Amprion", "TenneT", "50Hertz"],
            "electrical_capacity": [1.0, 3.0, 2.0, 2.0],
        }
    )
    assert build.tennet_shares(plants) == {"solar": 0.25, "onwind": 0.5}


def test_apply_profiles_rejects_missing_hours():
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "s", bus="b", carrier="solar", p_nom=1.0)
    cf = pd.read_csv(PROFILES, index_col=0, parse_dates=True)
    with pytest.raises(ValueError, match="lack"):
        apply.apply_profiles(n, cf)


def test_clipped_hours_counts_both_bounds():
    raw = pd.DataFrame({"solar": [-0.1, 0.5, 1.0, 1.2], "onwind": [0.0, 0.2, 0.3, 0.4]})
    assert build.clipped_hours(raw) == {
        "solar": {"below_0": 1, "above_1": 1},
        "onwind": {"below_0": 0, "above_1": 0},
    }

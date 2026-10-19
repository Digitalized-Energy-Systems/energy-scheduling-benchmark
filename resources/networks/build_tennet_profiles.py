"""Hourly 2019 capacity factors of solar, onshore and offshore wind in the TenneT
DE control area from measured production (decision Q23, TODO Phase 2c).

Production: ENTSO-E Transparency "Actual generation per production type", via
the Open Power System Data time-series package (2020-10-06), columns
``DE_tennet_{solar,wind_onshore,wind_offshore}_generation_actual`` (MW, UTC).

Installed capacity (the denominator):

* solar, onshore wind — TenneT's share of the OPSD renewable power plants
  register (2020-08-25; it ends in June 2018) times OPSD's hourly national
  capacity series for 2019 (``DE_solar_capacity``, ``DE_wind_onshore_capacity``),
  so the 2019 fleet growth is kept;
* offshore wind — all North Sea parks are connected by TenneT; grid-feeding
  North Sea capacity 5,236 MW at the end of 2018 and 6,440 MW at the end of
  2019 (Deutsche WindGuard, "Status des Offshore-Windenergieausbaus in
  Deutschland", 2018 and 2019), linear over 2019. OPSD's offshore capacities
  are below TenneT's own peak production and are not used.

Capacity factor = production / capacity, clipped to [0, 1]. Output: a CSV
(``utc_timestamp, solar, onwind, offwind``) plus a JSON sidecar with the input
files' SHA-256 and the method parameters.

Usage (from the repo root; the OPSD files are downloaded once to data/opsd/)::

    python energy-scheduling-benchmark/resources/networks/build_tennet_profiles.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OPSD_DIR = ROOT / "data" / "opsd"
OUT = Path(__file__).with_name("tennet_2019_capacity_factors.csv")

TIME_SERIES = "time_series_60min_singleindex.csv"
PLANTS = "renewable_power_plants_DE.csv"
URLS = {
    TIME_SERIES: "https://data.open-power-system-data.org/time_series/2020-10-06/"
    + TIME_SERIES,
    PLANTS: "https://data.open-power-system-data.org/renewable_power_plants/2020-08-25/"
    + PLANTS,
}
YEAR = 2019
#: North Sea grid-feeding offshore capacity (MW), Deutsche WindGuard status reports,
#: at the end of 2018 and of 2019 (i.e. the first instant of 2019 and of 2020).
NORTH_SEA_MW = {"2019-01-01": 5_236.0, "2020-01-01": 6_440.0}
GENERATION = {
    "solar": "DE_tennet_solar_generation_actual",
    "onwind": "DE_tennet_wind_onshore_generation_actual",
    "offwind": "DE_tennet_wind_offshore_generation_actual",
}
NATIONAL_CAPACITY = {"solar": "DE_solar_capacity", "onwind": "DE_wind_onshore_capacity"}
REGISTER_TECHNOLOGY = {
    "solar": ("Photovoltaics", "Photovoltaics ground"),
    "onwind": ("Onshore",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tennet_shares(plants: pd.DataFrame) -> dict[str, float]:
    """TenneT's share of the register's capacity per carrier."""
    shares = {}
    for carrier, techs in REGISTER_TECHNOLOGY.items():
        sub = plants[plants["technology"].isin(techs)]
        total = float(sub["electrical_capacity"].sum())
        tennet = float(sub.loc[sub["tso"] == "TenneT", "electrical_capacity"].sum())
        shares[carrier] = tennet / total
    return shares


def offshore_capacity(index: pd.DatetimeIndex) -> pd.Series:
    """North Sea capacity, linear from end-2018 to end-2019."""
    (t0, c0), (t1, c1) = (
        (pd.Timestamp(k, tz="UTC"), v) for k, v in NORTH_SEA_MW.items()
    )
    frac = ((index - t0) / (t1 - t0)).to_numpy(dtype=float)
    return pd.Series(c0 + np.clip(frac, 0.0, 1.0) * (c1 - c0), index=index)


def build(series: pd.DataFrame, shares: dict[str, float]) -> pd.DataFrame:
    """Capacity factors from one year of the OPSD time series (UTC index)."""
    cf = pd.DataFrame(index=series.index)
    for carrier in ("solar", "onwind"):
        capacity = series[NATIONAL_CAPACITY[carrier]].ffill().bfill() * shares[carrier]
        cf[carrier] = series[GENERATION[carrier]] / capacity
    cf["offwind"] = series[GENERATION["offwind"]] / offshore_capacity(series.index)
    return cf.clip(lower=0.0, upper=1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--opsd-dir", type=Path, default=OPSD_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    for name, url in URLS.items():
        if not (args.opsd_dir / name).is_file():
            parser.error(f"missing {args.opsd_dir / name}; download it from {url}")

    cols = ["utc_timestamp", *GENERATION.values(), *NATIONAL_CAPACITY.values()]
    ts = pd.read_csv(
        args.opsd_dir / TIME_SERIES, usecols=cols, parse_dates=["utc_timestamp"]
    )
    ts = ts.set_index("utc_timestamp")
    ts = ts[ts.index.year == YEAR]
    if len(ts) != 8760 or ts[list(GENERATION.values())].isna().any().any():
        raise SystemExit(f"expected 8760 complete hours of {YEAR}, got {len(ts)}")

    plants = pd.read_csv(
        args.opsd_dir / PLANTS,
        usecols=["electrical_capacity", "technology", "tso"],
        low_memory=False,
    )
    shares = tennet_shares(plants)
    cf = build(ts, shares)
    cf.index.name = "utc_timestamp"
    cf.round(6).to_csv(args.out)

    meta = {
        "year": YEAR,
        "method": __doc__.split("Usage")[0].strip(),
        "tennet_register_share": shares,
        "north_sea_capacity_mw": NORTH_SEA_MW,
        "sources": {
            name: {"url": URLS[name], "sha256": sha256(args.opsd_dir / name)}
            for name in URLS
        },
        "clipped_hours": {
            c: int((build(ts, shares)[c] >= 1.0).sum()) for c in cf.columns
        },
        "mean_capacity_factor": cf.mean().round(4).to_dict(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")
    print("TenneT register shares:", {k: round(v, 4) for k, v in shares.items()})
    print(
        "mean capacity factor:",
        meta["mean_capacity_factor"],
        "clipped hours:",
        meta["clipped_hours"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

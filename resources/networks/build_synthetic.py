"""Build ``synthetic_N.nc`` (raw input of the synthetic datasets): seeded candidates, keep the most
balanced one that is still varied (decided 2026-11-09, TODO Next #8a). Runs once per size; the
picked file is committed and the pick never reruns.

Candidate: N buses on a ring + N//2 random chords. Each bus has thermal and/or RES (solar or
onwind), storage on ~20 %, a load on ~70 %. Units are sized so each bus's price sensitivity
Σ 1/(2c₂) (the ``esb-prepare`` recipe: thermal p_nom/(2k·c₁), RES/storage p_nom/band) is
S = TARGET_S · U(0.8, 1.2), storage carrying STORAGE_SHARE of it. The RES fraction of S on
mixed buses is solved so RES is RES_SHARE of generator capacity. Load peak = thermal p_nom / 1.2.
Profiles: OPSD DE 2019 load / solar / onwind shapes with seeded per-bus noise.

Pick: candidates with thermal, RES, storage and load-free buses present; smallest max/min bus
sensitivity, checked with storage on and off (the worse of the two counts).

Input (not in git): ``data/opsd/time_series_60min_singleindex.csv``.

Usage (from the workspace root)::

    python energy-scheduling-benchmark/resources/networks/build_synthetic.py --buses 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from energy_scheduling_benchmark.datasets import CURVATURE_BAND, QUADRATIC_K

OPSD = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "opsd"
    / "time_series_60min_singleindex.csv"
)
TARGET_S = 100.0  # MW per €/MWh per bus
RES_SHARE = 0.3  # of generator capacity
STORAGE_SHARE = 0.2  # of a storage bus's sensitivity
P_STORAGE, P_LOAD = 0.2, 0.7
C1_RANGE = (20.0, 100.0)  # €/MWh
MARGIN = 1.2  # thermal p_nom / load peak
NOISE = 0.05  # std of the multiplicative hourly noise per bus


def draw(n_buses: int, seed: int) -> dict | None:
    """Candidate design as arrays (cheap); None if the RES share cannot be met."""
    # fewer buses cannot hold a ring + N//2 distinct chords (the chord loop would never end)
    assert n_buses >= 4, "need at least 4 buses"
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3, n_buses)  # 0 thermal only, 1 RES only, 2 mixed
    s = TARGET_S * rng.uniform(0.8, 1.2, n_buses)
    storage = rng.random(n_buses) < P_STORAGE
    s_gen = np.where(storage, 1 - STORAGE_SHARE, 1.0) * s
    c1 = rng.uniform(*C1_RANGE, n_buses)
    th_mw = 2 * QUADRATIC_K * c1  # thermal MW per unit of sensitivity
    # RES fraction f of s_gen on mixed buses so RES_cap = RES_SHARE · (RES_cap + thermal_cap)
    a = (s_gen * (kind == 1)).sum() * CURVATURE_BAND
    b = (s_gen * (kind == 2)).sum() * CURVATURE_BAND
    c = (s_gen * th_mw * (kind == 0)).sum()
    d = (s_gen * th_mw * (kind == 2)).sum()
    r = RES_SHARE / (1 - RES_SHARE)
    f = (r * (c + d) - a) / (b + r * d) if b + r * d > 0 else -1.0
    if not 0 < f < 1:
        return None
    res_s = np.select([kind == 1, kind == 2], [s_gen, f * s_gen], 0.0)
    lines = [(i, (i + 1) % n_buses) for i in range(n_buses)]
    while len(lines) < n_buses + n_buses // 2:
        i, j = sorted(rng.choice(n_buses, 2, replace=False))
        if (i, j) not in lines and (j, i) not in lines:
            lines.append((int(i), int(j)))
    return {
        "seed": seed,
        "s": s,
        "storage": storage,
        "c1": c1,
        "thermal_mw": (s_gen - res_s) * th_mw,
        "res_mw": res_s * CURVATURE_BAND,
        "res_carrier": rng.choice(["solar", "onwind"], n_buses),
        "storage_mw": np.where(storage, STORAGE_SHARE * s, 0.0) * CURVATURE_BAND,
        "load": rng.random(n_buses) < P_LOAD,
        "load_weight": rng.uniform(0.5, 1.5, n_buses),
        "lines": lines,
    }


def sensitivity(cand: dict, storage: bool) -> np.ndarray:
    """Bus Σ 1/(2c₂) under the esb-prepare recipe."""
    th = cand["thermal_mw"] / (2 * QUADRATIC_K * cand["c1"])
    other = cand["res_mw"] + (cand["storage_mw"] if storage else 0.0)
    return th + other / CURVATURE_BAND


def score(cand: dict) -> float:
    """Worse max/min sensitivity ratio of storage on/off; inf if not varied enough."""
    varied = (
        (cand["thermal_mw"] > 0).any()
        and (cand["res_mw"] > 0).any()
        and cand["storage"].any()
        and not cand["load"].all()
        and cand["load"].any()
    )
    if not varied:
        return np.inf
    on, off = sensitivity(cand, True), sensitivity(cand, False)
    return max(on.max() / on.min(), off.max() / off.min())


def build(cand: dict) -> pypsa.Network:
    rng = np.random.default_rng(
        [cand["seed"], 1]
    )  # profile noise, independent of the design
    cols = [
        "DE_load_actual_entsoe_transparency",
        "DE_solar_profile",
        "DE_wind_onshore_profile",
    ]
    ts = pd.read_csv(
        OPSD, usecols=["utc_timestamp", *cols], index_col=0, parse_dates=True
    )
    ts = ts.loc["2019"].interpolate().bfill()
    ts.index = ts.index.tz_localize(None)
    shape = {
        "load": ts[cols[0]] / ts[cols[0]].max(),
        "solar": ts[cols[1]],
        "onwind": ts[cols[2]],
    }

    def noisy(x: pd.Series) -> np.ndarray:
        return np.clip(x.to_numpy() * rng.normal(1, NOISE, len(x)), 0, 1)

    n = pypsa.Network()
    n.set_snapshots(ts.index)
    n.meta = {"seed": int(cand["seed"]), "builder": "build_synthetic.py"}
    buses = [f"bus{i}" for i in range(len(cand["s"]))]
    n.add("Bus", buses)
    for k, (i, j) in enumerate(cand["lines"]):
        n.add("Line", f"line{k}", bus0=buses[i], bus1=buses[j], x=0.1, s_nom=1e4)
    loads = {}
    for i, bus in enumerate(buses):
        if cand["thermal_mw"][i] > 0:
            n.add(
                "Generator",
                f"{bus} thermal",
                bus=bus,
                carrier="thermal",
                p_nom=cand["thermal_mw"][i],
                marginal_cost=cand["c1"][i],
            )
        if cand["res_mw"][i] > 0:
            car = str(cand["res_carrier"][i])  # numpy str_ breaks the NetCDF export
            n.add(
                "Generator",
                f"{bus} {car}",
                bus=bus,
                carrier=car,
                p_nom=cand["res_mw"][i],
                p_max_pu=noisy(shape[car]),
            )
        if cand["storage"][i]:
            # max_hours comes from the recipe (STORAGE_MAX_HOURS)
            n.add(
                "StorageUnit",
                f"{bus} battery",
                bus=bus,
                carrier="battery",
                p_nom=cand["storage_mw"][i],
            )
        if cand["load"][i]:
            loads[bus] = cand["load_weight"][i] * noisy(shape["load"])
    p_set = pd.DataFrame(loads, index=ts.index)
    p_set *= cand["thermal_mw"].sum() / MARGIN / p_set.sum(axis=1).max()
    n.add(
        "Load",
        [f"{b} load" for b in p_set],
        bus=list(p_set),
        p_set=p_set.set_axis([f"{b} load" for b in p_set], axis=1),
    )
    return n


def pick(n_buses: int, candidates: int) -> dict:
    cands = [c for seed in range(candidates) if (c := draw(n_buses, seed))]
    best = min(cands, key=score, default=None)
    if best is None or not np.isfinite(score(best)):
        raise SystemExit("no candidate meets the RES share and variety rules")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--buses", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=1000)
    args = parser.parse_args()
    best = pick(args.buses, args.candidates)
    n = build(best)
    out = Path(__file__).with_name(f"synthetic_{args.buses}.nc")
    n.export_to_netcdf(out)
    th, res = best["thermal_mw"].sum(), best["res_mw"].sum()
    on, off = (sensitivity(best, s) for s in (True, False))
    print(
        f"{out.name}: seed {best['seed']}, sensitivity max/min "
        f"{on.max() / on.min():.3f} (storage on), {off.max() / off.min():.3f} (off), "
        f"RES share {res / (th + res):.2f}, {len(n.buses)} buses, {len(n.lines)} lines, "
        f"{len(n.loads)} loads, {len(n.storage_units)} storage"
    )

"""esb-plot: figures from sweep records and benchmark runs (plan B4, D-157–D-163).

Every figure is written twice, ``<name>.html`` (interactive; plotly.js from the CDN) and
``<name>.pdf`` (kaleido + a Chrome: ``plotly_get_chrome`` installs one), next to its data. Needs
the optional extra ``[plot]``. Convergence curves run over iteration (D-162); gap per iteration =
trace ``cost`` / central ``total_cost`` − 1 (D-152), drawn as |gap| on a log axis — plotly has no
symlog — with the sign at the end marker (filled = above the optimum, open = below). The trace
cost is the units' cost only: in ``--grid-exchange`` runs the import added after the run is in
``total_cost`` (the gap bars) but not in the curve, so the curve can end below its bar.

Usage::

    esb-plot sweeps [ROOT]   # every sweep.json under ROOT (default outputs/sweeps) → <sweep>/plots/
                             # + ROOT/plots/overview_<algorithm>
    esb-plot runs [ROOT]     # esb-run folders under ROOT (default outputs) → ROOT/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .compare import compare, load_runs

COLORS = {"consensus": "#1f77b4", "admm": "#d62728"}
#: chosen set + this many best others in a sweep's PDF convergence figure (D-159)
PDF_BEST = 5


def color(algorithm: str) -> str:
    return COLORS.get(algorithm, "#7f7f7f")


#: sweep.csv columns that are not candidate parameters (esb-tune, D-148)
RESULT = {"candidate", "day", "status", "stop_reason", "iterations", "total_cost", "gap",
          "score", "sim_time_s", "wall_time_s", "folder",
          "solved_at", "imbalance"}  # fmt: skip  # 2026-11-23 D-196


def save(fig: go.Figure, path: Path, width=1000, height=600, pdf_only=None) -> None:
    """``<path>.html`` + ``<path>.pdf``; *pdf_only*: the trace ``meta`` values the PDF keeps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="plotly_white")
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    if pdf_only is not None:
        fig.data = [t for t in fig.data if t.meta in pdf_only]
    fig.write_image(path.with_suffix(".pdf"), width=width, height=height)


def gap_pct(cost, central_cost: float):
    """Relative gap in % (0 when equal, as ``tune.gap``; NaN when central costs 0)."""
    cost = np.asarray(cost, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(cost == central_cost, 0.0, (cost / central_cost - 1) * 100)
    return np.where(np.isfinite(g), g, np.nan)


def add_curves(fig, trace: pd.DataFrame, central_cost: float, col: int, **line) -> None:
    """|gap| (row 1) and imbalance (row 2) vs. iteration of one run into column *col*."""
    # float32: half the HTML size (a 32-candidate sweep at full resolution ~9 MB with float64)
    g = gap_pct(trace["cost"], central_cost).astype(np.float32)
    # the end point as its own marker: size-0 markers on every point bloated the PDF (9.8 MB)
    symbol = "circle" if g[-1] >= 0 else "circle-open"
    end = dict(
        mode="markers", marker=dict(symbol=symbol, size=8, color=line["line"]["color"])
    )
    # legend on the group's first curve: a run missing in the first column keeps its label
    first = line["legendgroup"] not in {t.legendgroup for t in fig.data}
    for row, y in ((1, np.abs(g)), (2, trace["imbalance_mwh"].to_numpy(np.float32))):
        fig.add_trace(
            go.Scatter(
                x=trace["iteration"], y=y, mode="lines", showlegend=row == 1 and first,
                customdata=g, hovertemplate="%{y:.4g}<br>gap %{customdata:+.4g} %", **line,
            ),
            row=row, col=col,
        )  # fmt: skip
        args = {k: v for k, v in line.items() if k != "line"}
        fig.add_trace(
            go.Scatter(
                x=trace["iteration"].iloc[-1:],
                y=y[-1:],
                showlegend=False,
                **end,
                **args,
            ),
            row=row,
            col=col,
        )


def convergence_layout(fig) -> None:
    fig.update_yaxes(type="log", exponentformat="power")
    fig.update_yaxes(title_text="|gap| [%]", row=1, col=1)
    fig.update_yaxes(title_text="imbalance [MWh]", row=2, col=1)
    fig.update_xaxes(title_text="iteration", row=2)  # D-162


# ── sweeps ────────────────────────────────────────────────────────────────────────────────────


def candidates(table: pd.DataFrame, record: dict) -> pd.DataFrame:
    """One row per candidate: its parameters, ``label``, ``status`` (ok / failed / pruned),
    ``key`` (its worst window's score, as esb-tune, None unless ok on every day)."""
    # a variant's own parameters (κ, γ) come after the result fields when an earlier variant
    # lacks them (pandas column order = first row's keys)
    params = [c for c in table.columns if c not in RESULT]
    rows = []
    for i, runs in table.groupby("candidate"):
        first = runs.iloc[0]
        values = {p: first[p] for p in params if pd.notna(first[p])}
        status = (
            "failed" if (runs.status == "failed").any()
            else "pruned" if (runs.status == "skipped").any() else "ok"
        )  # fmt: skip
        key = None
        if status == "ok":
            key = runs.score.max()
        label = ", ".join(
            f"{v}"
            if p == "steps"
            else f"{p}={v:.4g}"
            if isinstance(v, float)
            else f"{p}={v}"
            for p, v in values.items()
            if p != "agent_scale"
        )
        rows.append(
            {"candidate": i, **values, "label": label, "status": status, "key": key}
        )
    return pd.DataFrame(rows).set_index("candidate", drop=False)


def chosen(record: dict) -> int | None:
    """Index of the set esb-tune picked."""
    if "chosen_candidate" in record:  # 2026-11-18 D-182
        return record["chosen_candidate"]
    best = record["best"]  # records before D-182: the lowest rank key
    return min(best, key=lambda b: b["rank"])["candidate"] if best else None


def sweep_convergence(store: Path, table: pd.DataFrame, record: dict, cands) -> None:
    """D-159: HTML every candidate that ran (failed dashed), PDF chosen + the PDF_BEST best."""
    days = list(record["central"])
    best = chosen(record)
    fig = make_subplots(
        rows=2, cols=len(days), shared_xaxes=True, column_titles=[d[:10] for d in days],
        vertical_spacing=0.06, horizontal_spacing=0.03,
    )  # fmt: skip
    colors = px.colors.qualitative.Dark24
    for _, r in table[table.status != "skipped"].iterrows():
        c = cands.loc[r.candidate]
        trace = pd.read_csv(store / r.folder / "trace.csv")
        add_curves(
            fig, trace, record["central"][r.day]["total_cost"], days.index(r.day) + 1,
            name=f"{'★ ' if r.candidate == best else ''}{c.label}",
            legendgroup=str(r.candidate),
            line=dict(
                color=colors[r.candidate % len(colors)],
                width=3 if r.candidate == best else 1,
                dash="dash" if c.status == "failed" else "solid",
            ),
            meta=int(r.candidate),
        )  # fmt: skip
    a = record["args"]
    fig.update_layout(
        title=f"{a['algorithm']} sweep — {a['network']}, {a['topology']}, "
        f"{'' if a['storage'] else 'no '}storage"
    )
    convergence_layout(fig)
    ranked = sorted(cands.key.dropna().items(), key=lambda ik: (ik[1], ik[0]))
    keep = {i for i, _ in ranked[: PDF_BEST + 1]} | (
        {best} if best is not None else set()
    )
    save(fig, store / "plots" / "convergence", 1400, 700, pdf_only=keep)


def sweep_grid(store: Path, table: pd.DataFrame, record: dict, cands) -> None:
    """D-160: consensus = heatmap of the worst-window score per step variant; admm = per day
    iterations and gap vs. ρ; deed_admm = the same vs. γ, one column per β (D-180, D-185);
    diffusion = the same vs. ε (S8a)
    Pruned days have no point."""
    best = chosen(record)
    if "rho" in cands or "beta" in cands or "eps_step" in cands:
        x = next(c for c in ("rho", "eps_step", "gamma") if c in cands)
        betas = sorted(cands.beta.unique()) if "beta" in cands else [None]
        ok = table[table.status == "ok"].sort_values(x)
        days = sorted(ok.day.str[:10].unique())
        fig = make_subplots(
            rows=2, cols=len(betas), shared_xaxes=True, vertical_spacing=0.06,
            column_titles=None if betas == [None] else [f"β = {b:.4g}" for b in betas],
        )  # fmt: skip
        for col, b in enumerate(betas, 1):
            sub = ok if b is None else ok[ok.beta == b]
            for day, runs in sub.groupby(sub.day.str[:10]):
                for row, y in ((1, runs.iterations), (2, runs.gap * 100)):
                    fig.add_trace(
                        go.Scatter(x=runs[x], y=y, name=day, legendgroup=day,
                                   mode="lines+markers", showlegend=row == 1 and col == 1,
                                   line_color=px.colors.qualitative.Plotly[days.index(day)]),
                        row=row, col=col,
                    )  # fmt: skip
        if best is not None:  # the chosen ρ / γ, in the chosen β's column
            c = 1 if betas == [None] else betas.index(cands.loc[best, "beta"]) + 1
            fig.add_vline(x=cands.loc[best, x], line_dash="dot", col=c)
        fig.update_xaxes(type="log")
        fig.update_xaxes(title_text={"rho": "ρ", "eps_step": "ε"}.get(x, "γ"), row=2)
        fig.update_yaxes(title_text="iterations", type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1)
        fig.update_yaxes(title_text="gap [%]", row=2, col=1)
    else:
        panels = []  # (title, candidates, x column, x title, y column)
        for steps, cs in cands.groupby("steps"):
            if steps == "constant":
                cs = cs.assign(beta=cs.kappa * cs.gamma)
                panels.append(("constant", cs, "beta", "β = κγ", "kappa"))
            else:
                for e, ce in cs.groupby("alpha_exp"):
                    panels.append(
                        (f"decaying, α exp {e:g}", ce, "beta0", "β₀", "alpha0")
                    )
        fig = make_subplots(
            rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels],
            horizontal_spacing=0.08,
        )  # fmt: skip
        gaps = (
            table.dropna(subset=["gap"])
            .groupby("candidate")
            .apply(
                lambda r: "<br>".join(f"{d}: {g:+.3%}" for d, g in zip(r.day, r.gap)),
                include_groups=False,
            )
        )
        mark = {"failed": "×", "pruned": "·", "ok": ""}
        for col, (_, cs, x, xtitle, y) in enumerate(panels, 1):
            z = np.log10(cs.key.astype(float))
            text = [
                "★" if i == best else mark[s] for i, s in zip(cs.candidate, cs.status)
            ]
            fig.add_trace(
                go.Heatmap(
                    x=[f"{v:.3g}" for v in cs[x]], y=[f"{v:.3g}" for v in cs[y]],
                    z=z, text=text, texttemplate="%{text}", textfont_size=18,
                    customdata=[gaps.get(i, "") for i in cs.candidate],
                    hovertemplate="%{customdata}<extra>log10 score %{z:.2f}</extra>",
                    coloraxis="coloraxis",
                ),
                row=1, col=col,
            )  # fmt: skip
            fig.update_xaxes(title_text=xtitle, type="category", row=1, col=col)
            fig.update_yaxes(title_text=y, type="category", row=1, col=col)
        fig.update_layout(
            coloraxis=dict(colorscale="Viridis_r", colorbar_title="log10 score")
        )
    a = record["args"]
    fig.update_layout(
        title=f"{a['algorithm']} sweep result — {a['network']}, {a['topology']}, "
        f"{'' if a['storage'] else 'no '}storage "
        + (
            f"(dotted: chosen {'ρ' if 'rho' in cands else 'γ'})"
            if "rho" in cands or "beta" in cands
            else "(★ chosen, × failed, · pruned)"
        )
    )
    save(fig, store / "plots" / "grid", width=1100, height=550)


def overview(root: Path, summary: pd.DataFrame) -> None:
    """D-160: per algorithm, the chosen set's mean tuning-day gap per network × topology."""
    for alg, s in summary.groupby("algorithm"):
        fig = px.imshow(
            s.pivot_table(
                index="network", columns=["storage", "topology"], values="gap_pct",
                dropna=False,
            ).rename(columns={True: "storage", False: "no storage"}, level=0)
            .pipe(lambda p: p.set_axis([f"{a} {b}" for a, b in p.columns], axis=1)),
            text_auto=".3g", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            aspect="auto", labels={"color": "gap [%]"},
        )  # fmt: skip
        fig.update_layout(
            title=f"{alg} sweeps — chosen set, mean 24 h gap on the tuning days"
        )
        save(fig, root / "plots" / f"overview_{alg}", width=900, height=500)


def plot_sweeps(root: Path) -> None:
    summary = []
    for path in sorted(root.rglob("sweep.json")):
        store, record = path.parent, json.loads(path.read_text())
        table = pd.read_csv(store / "sweep.csv")
        cands = candidates(table, record)
        sweep_convergence(store, table, record, cands)
        sweep_grid(store, table, record, cands)
        best, a = chosen(record), record["args"]
        summary.append(
            {k: a[k] for k in ("algorithm", "network", "topology", "storage")}
            | {"gap_pct": None if best is None
               else table[table.candidate == best].gap.mean() * 100}
        )  # fmt: skip
        print(f"{store}/plots/")
    if not summary:
        raise SystemExit(f"no sweep.json under {root}")
    overview(root, pd.DataFrame(summary))
    print(f"{root}/plots/overview_*")


# ── runs ──────────────────────────────────────────────────────────────────────────────────────


def plot_runs(root: Path) -> None:
    """D-161: gap and effort per network × standard day, convergence per network/day."""
    runs = load_runs(root)
    if runs.empty:
        raise SystemExit(f"no runs under {root}/<network>/<algorithm>/<run>/")
    rows, _ = compare(runs)
    # shortcut: lossless runs only; the loss dimension gets its own figure at S5 step 3
    rows = rows[rows.loss_rate == 0].copy()
    if rows.empty:  # e.g. an S5 batch folder: say so instead of writing empty figures
        print(f"no lossless runs with a central run under {root}: no figures")
        return
    rows["label"] = rows.algorithm + rows.variant.map(lambda v: f"[{v}]" if v else "")
    rows["case"] = (
        rows.network + " " + pd.to_datetime(rows.start).dt.strftime("%m-%d")
        + rows.storage.map({True: "", False: " no st."})
    )  # fmt: skip
    rows = rows.sort_values(["network", "start", "storage", "label"])
    common = dict(
        x="case", color="label", barmode="group", facet_col="topology",
        hover_data=["stop_reason", "run"],
        # colour by algorithm (a label adds the tuning route, e.g. "admm[sweep]")
        color_discrete_map={r.label: color(r.algorithm) for r in rows.itertuples()},
    )  # fmt: skip
    fig = px.bar(rows, y="gap_pct", **common, labels={"gap_pct": "gap [%]"})
    fig.update_layout(title="cost gap vs. central dispatch")
    save(fig, root / "plots" / "gap", width=1400, height=500)
    effort = rows.melt(
        id_vars=["case", "label", "topology", "stop_reason", "run"],
        value_vars=["iterations", "messages_sent"], var_name="metric", value_name="n",
    )  # fmt: skip
    fig = px.bar(effort, y="n", facet_row="metric", **common)
    fig.update_yaxes(type="log", matches=None, title_text="")
    fig.update_layout(title="effort: iterations and messages sent")
    save(fig, root / "plots" / "effort", width=1400, height=700)
    for (network, start, storage), case in rows.groupby(
        ["network", "start", "storage"]
    ):
        topologies = sorted(case.topology.unique())
        fig = make_subplots(
            rows=2, cols=len(topologies), shared_xaxes=True, column_titles=topologies,
            vertical_spacing=0.06,
        )  # fmt: skip
        for _, r in case.iterrows():
            trace = Path(r.run) / "trace.csv"
            if trace.exists():  # non-iterative algorithms (dummy) have none
                add_curves(
                    fig, pd.read_csv(trace), r.central_cost,
                    topologies.index(r.topology) + 1, name=r.label, legendgroup=r.label,
                    line=dict(color=color(r.algorithm)),
                )  # fmt: skip
        day = pd.Timestamp(start).strftime("%Y%m%d")
        st = "storage" if storage else "nostorage"
        fig.update_layout(title=f"convergence — {network}, {start}, {st}")
        convergence_layout(fig)
        save(fig, root / "plots" / f"convergence_{network}_{day}_{st}", width=1200)
    print(f"{root}/plots/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("sweeps", help="esb-tune records").add_argument(
        "root", nargs="?", default="outputs/sweeps"
    )
    sub.add_parser("runs", help="esb-run folders").add_argument(
        "root", nargs="?", default="outputs"
    )
    args = parser.parse_args(argv)
    (plot_sweeps if args.what == "sweeps" else plot_runs)(Path(args.root))


if __name__ == "__main__":
    main()

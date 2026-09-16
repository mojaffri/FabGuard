"""Run all experiments and save auditable synthetic evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import matching, optimize, spc
from .fleet import reliability, simulate
from .maintenance import evaluate, labels
from .model import Recipe, wafer


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def chart_study(seed=7):
    rng = np.random.default_rng(seed)
    baseline = np.array([[wafer(rng=rng)["depth_nm"] for _ in range(5)] for _ in range(40)])
    b = spc.fit(baseline)
    charts, summaries, raw = [], [], []
    for scenario in [
        "stable",
        "wall_drift",
        "rf_loss",
        "vacuum_leak",
        "flow_restriction",
        "sensor_bias",
    ]:
        # Matched random noise across scenarios isolates each injection's effect.
        noise = np.random.default_rng(seed + 1000)
        values = []
        for lot in range(100):
            age = max(0, lot - 29) * 0.012 if scenario == "wall_drift" else 0
            fault = scenario if lot >= 30 and scenario not in {"stable", "wall_drift"} else "none"
            rows = [wafer(age=age, fault=fault, rng=noise) for _ in range(5)]
            values.append([r["depth_nm"] for r in rows])
            raw.extend(
                dict(scenario=scenario, lot=lot, wafer=j, **row) for j, row in enumerate(rows)
            )
        c = spc.monitor(b, values)
        c["lot"] = np.arange(100)
        c["scenario"] = scenario
        charts.append(c)
        summary = dict(seed=seed, scenario=scenario, baseline_signals=b.phase1_signals)
        for kind in ["xbar", "r", "ewma", "cusum"]:
            alarms = c[f"{kind}_alarm"].to_numpy()
            hits = np.flatnonzero(alarms[30:])
            summary[f"{kind}_delay_lots"] = int(hits[0]) if len(hits) else None
            summary[f"{kind}_predrift_alarms"] = int(alarms[:30].sum())
            summary[f"{kind}_stable_alarms"] = int(alarms.sum()) if scenario == "stable" else None
        summaries.append(summary)
    return (
        baseline,
        pd.concat(charts, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(raw),
    )


def figures(out, charts, match, history, fleet_metrics):
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f5f7fa",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )
    c = charts.loc[charts.scenario == "wall_drift"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, value, lower, upper, title in [
        (axes[0, 0], "xbar", "xbar_lcl", "xbar_ucl", "Subgroup mean / nm"),
        (axes[0, 1], "range", "r_lcl", "r_ucl", "Within-subgroup range / nm"),
        (axes[1, 0], "ewma", "ewma_lcl", "ewma_ucl", "EWMA / nm"),
    ]:
        ax.plot(c.lot, c[value], color="#126b78")
        ax.plot(c.lot, c[lower], "--", color="#b65333", linewidth=1)
        ax.plot(c.lot, c[upper], "--", color="#b65333", linewidth=1)
        ax.axvline(30, color="#65758a", linestyle=":")
        ax.set(title=title, xlabel="Phase-II lot (five wafers)")
    axes[1, 1].plot(c.lot, c.cusum_plus, label="C+", color="#126b78")
    axes[1, 1].plot(c.lot, c.cusum_minus, label="C−", color="#b65333")
    axes[1, 1].axhline(5, linestyle="--", color="#65758a")
    axes[1, 1].set(title="Standardized CUSUM", xlabel="Phase-II lot")
    axes[1, 1].legend()
    fig.suptitle("FabGuard | Synthetic wall drift · frozen limits · onset at lot 30")
    fig.savefig(out / "spc.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    positions = np.arange(len(match))
    axes[0].bar(positions - 0.18, match.before_delta_nm, 0.36, label="Before", color="#b65333")
    axes[0].bar(positions + 0.18, match.after_delta_nm, 0.36, label="After", color="#126b78")
    axes[0].set(
        xticks=positions,
        xticklabels=match.tool,
        title="Independent matching confirmation",
        ylabel="Offset from ETCH-01 / nm",
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    for method, frame in history.groupby("method"):
        mean = frame.groupby("evaluation").best_loss.mean()
        axes[1].plot(mean.index, mean, label=method)
    axes[1].set(
        title="Equal-budget search (mean over seeds)",
        xlabel="Recipe evaluations",
        ylabel="Best synthetic loss ↓",
    )
    axes[1].legend()
    axes[2].bar(fleet_metrics.tool, fleet_metrics.OEE * 100, color="#126b78")
    axes[2].set(title="Fleet OEE", ylabel="Percent", ylim=(0, 100))
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle("Simulation results — no physical wafers or production equipment")
    fig.savefig(out / "overview.png", dpi=160)
    plt.close(fig)


def run(out, seed=7, quick=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("Simulating four-tool fleet...", flush=True)
    wafers, sensors, events = simulate(seed)
    for name, data in [("wafers", wafers), ("sensors", sensors), ("events", events)]:
        data.to_csv(out / f"{name}.csv", index=False)
    fleet_metrics = reliability(events)
    fleet_metrics.to_csv(out / "reliability.csv", index=False)
    baseline, charts, detection, chart_wafers = chart_study(seed)
    pd.DataFrame(baseline).to_csv(out / "spc_baseline.csv", index=False)
    charts.to_csv(out / "charts.csv", index=False)
    chart_wafers.to_csv(out / "challenge_wafers.csv", index=False)
    repeated = [detection]
    for extra in range(1, 5 if quick else 30):
        repeated.append(chart_study(seed + extra)[2])
    detections = pd.concat(repeated, ignore_index=True)
    detections.to_csv(out / "detection_trials.csv", index=False)
    calibration, confirmation, match = matching.experiment(seed)
    calibration.to_csv(out / "matching_calibration.csv", index=False)
    confirmation.to_csv(out / "matching_confirmation.csv", index=False)
    match.to_csv(out / "matching.csv", index=False)
    print("Training and evaluating maintenance forecasts on disjoint campaigns...", flush=True)
    blocks = []
    for campaign in range(seed + 100, seed + 114):
        _, s, _ = simulate(campaign, lots=180 if quick else 240)
        blocks.append(labels(s))
    data = pd.concat(blocks, ignore_index=True)
    train = data.loc[data.campaign < seed + 108]
    validation = data.loc[(data.campaign >= seed + 108) & (data.campaign < seed + 110)]
    test = data.loc[data.campaign >= seed + 110]
    predictions, maintenance = evaluate(train, validation, test, seed)
    data.to_csv(out / "maintenance_dataset.csv", index=False)
    predictions.to_csv(out / "maintenance_predictions.csv", index=False)
    print("Comparing Bayesian and random recipe search...", flush=True)
    histories, opt_results = [], []
    for opt_seed in [seed, seed + 10, seed + 20]:
        history, recipes = optimize.search(
            opt_seed, budget=16 if quick else 30, initial=8 if quick else 10
        )
        histories.append(history)
        for method, recipe in {"nominal": Recipe(), **recipes}.items():
            opt_results.append(
                dict(
                    seed=opt_seed,
                    method=method,
                    loss=optimize.objective(recipe),
                    **optimize.confirm(recipe, seed=opt_seed + 9900),
                    **asdict(recipe),
                )
            )
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(out / "optimization_history.csv", index=False)
    optimization = pd.DataFrame(opt_results)
    optimization.to_csv(out / "optimization_confirmation.csv", index=False)
    metrics = dict(
        simulation_only=True,
        seed=seed,
        quick=quick,
        fleet_lots=240,
        wafers=len(wafers),
        capability=spc.capability(baseline),
        maintenance=maintenance,
        matching_before_span_nm=float(match.before_mean_nm.max() - match.before_mean_nm.min()),
        matching_after_span_nm=float(match.after_mean_nm.max() - match.after_mean_nm.min()),
        optimization=optimization.to_dict("records"),
    )
    save_json(out / "metrics.json", metrics)
    figures(out, charts, match, history, fleet_metrics)
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        list((root / "src").rglob("*.py"))
        + list((root / "tests").rglob("*.py"))
        + list((root / "scripts").rglob("*.py"))
        + [
            root / "dashboard.py",
            root / "pyproject.toml",
            root / "requirements-reference.txt",
            root / ".streamlit/config.toml",
        ]
    )
    save_json(
        out / "manifest.json",
        dict(
            seed=seed,
            simulation_only=True,
            python=platform.python_version(),
            command=f"python -m fabguard.experiments --seed {seed}" + (" --quick" if quick else ""),
            versions={
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "streamlit",
                    "plotly",
                ]
            },
            source_sha256={
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            data_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.glob("*.csv"))
            },
        ),
    )
    drift = detections.loc[detections.scenario == "wall_drift"]
    stable = detections.loc[detections.scenario == "stable"]
    lines = [
        "# FabGuard reference experiment",
        "",
        "All results are synthetic. No real fab data or physical experiments.",
        "",
        f"Seed {seed}; {len(wafers):,} fleet wafers; {len(drift)} independent SPC trials per scenario.",
        "",
        "![Equipment analytics](overview.png)",
        "",
        "## Drift detection",
        "",
        "Delays count lots after onset (zero means the first affected lot). Undetected trials remain censored; false alarms are retained.",
        "",
        "| Chart | Median detected delay / lots | Detected trials | Stable point alarm fraction |",
        "|---|---:|---:|---:|",
    ]
    for kind in ["xbar", "r", "ewma", "cusum"]:
        delay = drift[f"{kind}_delay_lots"]
        lines.append(
            f"| {kind} | {delay.median():.1f} | {delay.notna().sum()}/{len(delay)} | {stable[f'{kind}_stable_alarms'].sum() / (100 * len(stable)):.3%} |"
        )
    lines += [
        "",
        "CUSUM is not reset after an alarm; point alarm fractions include persistent excursions. These are not average-run-length estimates or familywise error rates. The wall challenge changes the mean, not within-group dispersion: R-chart flags here are incidental false alarms, not evidence of sensitivity to wall drift. Delay comparisons are not adjusted to equal false-alarm rates.",
        "",
        "![SPC](spc.png)",
        "",
        "## Tool matching",
        "",
        f"Across-tool mean depth span: {metrics['matching_before_span_nm']:.2f} → {metrics['matching_after_span_nm']:.2f} nm on independent confirmation groups (60 wafers per tool and condition).",
        "Offsets are estimated only from calibration wafers. Clean-tool time compensation does not fix uniformity, sensor faults, or aging. See matching.csv for Welch intervals and ±3 nm equivalence checks.",
        "",
        "## Maintenance forecast",
        "",
        "| Model | Test PR-AUC | Brier score | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["sensor_forest", "age_only", "constant_prevalence"]:
        m = maintenance[name]
        lines.append(
            f"| {name} | {m['PR_AUC']:.3f} | {m['Brier']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |"
        )
    lines += [
        "",
        f"Test prevalence: {maintenance['sensor_forest']['prevalence']:.1%}; campaign-bootstrap PR-AUC interval: {maintenance['sensor_forest']['PR_AUC_campaign_bootstrap_95']}. Only four test campaigns; uncertainty estimates are coarse.",
        "The label excludes current failures and censored futures. No latent wall condition or future event is a feature. Forecast accuracy is not evidence of maintenance cost savings; no learned maintenance policy is deployed.",
        "",
        "## Recipe search",
        "",
        "| Method | Mean loss ↓ | Confirmation good fraction | Wafers/h | Wear index ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, frame in optimization.groupby("method"):
        lines.append(
            f"| {name} | {frame.loss.mean():.3f} | {frame.good_fraction.mean():.1%} | {frame.throughput_wafers_h.mean():.2f} | {frame.wear_index.mean():.3f} |"
        )
    lines += [
        "",
        "Three search seeds; equal evaluation budget and initial design. Confirmation uses new wafer noise and wall conditions. The loss is a chosen weighted objective with soft penalties, not a validated cost model or a guarantee of feasibility.",
        "",
        "## Reliability accounting",
        "",
        "| Tool | Failures | MTBF / h | MTTR / h | Availability | Quality | OEE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r.tool} | {r.failures} | {r.MTBF_h:.2f} | {r.MTTR_h:.2f} | {r.availability:.1%} | {r.quality:.1%} | {r.OEE:.1%} |"
            for r in fleet_metrics.itertuples()
        ),
        "MTBF is observed operating exposure / corrective failures. MTTR is completed repair time / repairs. Scheduled cleans count against availability but not corrective MTTR. All event durations, including final service, are inside the recorded campaign horizon. OEE = availability × performance × good fraction; the ideal cycle is 135 s/wafer.",
        "",
        "Source/dependency manifest, raw CSVs, frozen baselines, predictions and optimizer histories accompany this report.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evidence saved to {out.resolve()}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    run(args.out, args.seed, args.quick)


if __name__ == "__main__":
    main()

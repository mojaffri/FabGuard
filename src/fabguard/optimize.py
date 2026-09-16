"""Bounded Gaussian-process expected-improvement search and equal-budget random baseline."""

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .model import TOOLS, Recipe, response, wafer

LOW = np.array([550.0, 20.0, 30.0, 90.0])
HIGH = np.array([950.0, 50.0, 75.0, 160.0])


def decode(x):
    power, pressure, flow, duration = LOW + np.asarray(x) * (HIGH - LOW)
    return Recipe(power_w=power, pressure_mtorr=pressure, flow_sccm=flow, time_s=duration)


def objective(recipe):
    # Same predefined reference tool and wall-condition ensemble for every evaluation.
    rows = [response(recipe, tool, age) for tool in TOOLS for age in (0.0, 0.4, 0.8)]
    losses = []
    for r in rows:
        violation = (
            max(0, abs(r["depth_nm"] - 500) - 20) / 20
            + max(0, r["nonuniformity_pct"] - 3)
            + max(0, abs(r["cd_error_nm"]) - 5) / 5
        )
        losses.append(
            2 * ((r["depth_nm"] - 500) / 20) ** 2
            + 0.4 * (r["nonuniformity_pct"] / 3) ** 2
            + 0.8 * r["defect_probability"] / 0.05
            + 0.6 * r["cycle_s"] / 165
            + 0.3 * r["wear_index"] / 1.2
            + 8 * violation**2
        )
    return float(np.mean(losses))


def expected_improvement(mean, std, best):
    safe = np.maximum(std, 1e-12)
    improvement = best - mean - 0.01
    z = improvement / safe
    return np.where(std > 1e-12, improvement * norm.cdf(z) + safe * norm.pdf(z), 0)


def search(seed=7, budget=30, initial=10):
    if initial < 4 or budget < initial + 1:
        raise ValueError("Need >=4 initial evaluations and a larger budget")
    first = qmc.LatinHypercube(4, seed=seed).random(initial)
    # Include the nominal recipe in both budgets.
    first[0] = (np.array([750.0, 35.0, 50.0, 120.0]) - LOW) / (HIGH - LOW)
    records, best_recipes = [], {}
    for method in ["bayesian", "random"]:
        rng = np.random.default_rng(seed + 100)
        xs, ys = list(first.copy()), [objective(decode(x)) for x in first]
        for step in range(initial, budget):
            if method == "random":
                x = rng.random(4)
            else:
                # Fixed prior smoothness is transparent and avoids fitting a small-budget kernel to noise.
                gp = GaussianProcessRegressor(
                    kernel=ConstantKernel(1, "fixed") * Matern(0.35, "fixed", nu=2.5),
                    alpha=1e-6,
                    normalize_y=True,
                    optimizer=None,
                )
                gp.fit(np.array(xs), ys)
                candidates = rng.random((768, 4))
                mean, std = gp.predict(candidates, return_std=True)
                x = candidates[int(np.argmax(expected_improvement(mean, std, min(ys))))]
            xs.append(x)
            ys.append(objective(decode(x)))
        for i, (x, loss) in enumerate(zip(xs, ys)):
            records.append(
                dict(
                    seed=seed,
                    method=method,
                    evaluation=i + 1,
                    loss=loss,
                    best_loss=min(ys[: i + 1]),
                    **asdict(decode(x)),
                )
            )
        best_recipes[method] = decode(xs[int(np.argmin(ys))])
    return pd.DataFrame(records), best_recipes


def confirm(recipe, seed=9907, n=300):
    rng = np.random.default_rng(seed)
    rows = [wafer(recipe, TOOLS[i % 4], float(rng.uniform(0, 0.9)), rng=rng) for i in range(n)]
    data = pd.DataFrame(rows)
    return dict(
        n=n,
        good_fraction=float(data.good.mean()),
        depth_RMSE_nm=float(np.sqrt(np.mean((data.depth_nm - 500) ** 2))),
        nonuniformity_pct=float(data.nonuniformity_pct.mean()),
        particle_fraction=float(data.particle_defect.mean()),
        throughput_wafers_h=3600 / (recipe.time_s + 45),
        wear_index=float(data.wear_index.mean()),
    )

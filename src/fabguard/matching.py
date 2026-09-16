"""Estimate time offsets on calibration wafers, confirm on new independent wafers."""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import t

from .model import TOOLS, Recipe, wafer


def experiment(seed=7, n=60):
    if n < 5:
        raise ValueError("Need at least five wafers per group")
    rng = np.random.default_rng(seed)
    calibration, confirmation, results = [], [], []
    means = {}
    for tool in TOOLS:
        values = [wafer(tool=tool, rng=rng)["depth_nm"] for _ in range(n)]
        means[tool.name] = float(np.mean(values))
        calibration.extend(dict(tool=tool.name, depth_nm=v) for v in values)
    reference = means[TOOLS[0].name]
    for tool in TOOLS:
        corrected = replace(
            Recipe(), time_s=float(np.clip(120 * reference / means[tool.name], 90, 160))
        )
        for group, recipe in [("before", Recipe()), ("after", corrected)]:
            values = [wafer(recipe, tool, rng=rng)["depth_nm"] for _ in range(n)]
            confirmation.extend(dict(tool=tool.name, group=group, depth_nm=v) for v in values)
        results.append(
            dict(
                tool=tool.name,
                calibration_mean_nm=means[tool.name],
                recommended_time_s=corrected.time_s,
                offset_s=corrected.time_s - 120,
            )
        )
    data = pd.DataFrame(confirmation)
    for row in results:
        for group in ["before", "after"]:
            x = data.loc[(data.tool == row["tool"]) & (data.group == group), "depth_nm"]
            ref = data.loc[(data.tool == TOOLS[0].name) & (data.group == group), "depth_nm"]
            delta = float(x.mean() - ref.mean())
            a, b = x.var(ddof=1) / n, ref.var(ddof=1) / n
            df = (a + b) ** 2 / (a * a / (n - 1) + b * b / (n - 1))
            width = float(t.ppf(0.975, df) * np.sqrt(a + b))
            row[f"{group}_mean_nm"] = float(x.mean())
            row[f"{group}_delta_nm"] = delta
            row[f"{group}_ci_lo_nm"] = delta - width if row["tool"] != TOOLS[0].name else 0.0
            row[f"{group}_ci_hi_nm"] = delta + width if row["tool"] != TOOLS[0].name else 0.0
            # Equivalence criterion, not a failure to reject a difference.
            row[f"{group}_equivalent_3nm"] = (
                bool(abs(delta) + width < 3) if row["tool"] != TOOLS[0].name else True
            )
    return pd.DataFrame(calibration), data, pd.DataFrame(results)

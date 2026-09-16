"""Frozen Phase-I limits for rational subgroups of five consecutive wafers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def subgroups(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite rows of five consecutive same-tool, same-recipe wafers")
    return x


@dataclass(frozen=True)
class Baseline:
    mean: float
    rbar: float
    sigma: float
    phase1_signals: int
    lag1: float | None


def fit(values):
    x = subgroups(values)
    if len(x) < 25:
        raise ValueError("At least 25 Phase-I subgroups required")
    mean, rbar = float(x.mean()), float(np.ptp(x, axis=1).mean())
    if rbar <= 0:
        raise ValueError("Zero within-subgroup variation")
    signals = (abs(x.mean(axis=1) - mean) > 0.577 * rbar) | (np.ptp(x, axis=1) > 2.114 * rbar)
    means = x.mean(axis=1)
    lag1 = (
        float(np.corrcoef(means[:-1], means[1:])[0, 1])
        if min(means[:-1].std(), means[1:].std()) > 0
        else None
    )
    return Baseline(
        mean,
        rbar,
        rbar / 2.326,
        int(signals.sum()),
        lag1,
    )


def monitor(baseline, values, lam=0.2, k=0.5, h=5.0):
    x = subgroups(values)
    if not np.isfinite([lam, k, h]).all() or not 0 < lam <= 1 or k <= 0 or h <= 0:
        raise ValueError("Invalid chart parameters")
    z, plus, minus, rows = baseline.mean, 0.0, 0.0, []
    se = baseline.sigma / np.sqrt(5)
    for i, group in enumerate(x, 1):
        mean, spread = float(group.mean()), float(np.ptp(group))
        z = lam * mean + (1 - lam) * z
        ew = 3 * se * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
        standardized = (mean - baseline.mean) / se
        plus, minus = max(0, plus + standardized - k), max(0, minus - standardized - k)
        rows.append(
            dict(
                subgroup=i,
                xbar=mean,
                range=spread,
                ewma=z,
                xbar_lcl=baseline.mean - 0.577 * baseline.rbar,
                xbar_ucl=baseline.mean + 0.577 * baseline.rbar,
                r_lcl=0.0,
                r_ucl=2.114 * baseline.rbar,
                ewma_lcl=baseline.mean - ew,
                ewma_ucl=baseline.mean + ew,
                cusum_plus=plus,
                cusum_minus=minus,
                cusum_h=h,
                xbar_alarm=bool(abs(mean - baseline.mean) > 0.577 * baseline.rbar),
                r_alarm=bool(spread > 2.114 * baseline.rbar),
                ewma_alarm=bool(abs(z - baseline.mean) > ew),
                cusum_alarm=bool(max(plus, minus) > h),
            )
        )
    return pd.DataFrame(rows)


def capability(values, lsl=480, usl=520):
    x = subgroups(values)
    b = fit(x)
    if not np.isfinite([lsl, usl]).all() or usl <= lsl:
        raise ValueError("Invalid specification limits")
    overall = float(x.std(ddof=1))
    return dict(
        Cp=(usl - lsl) / (6 * b.sigma),
        Cpk=min(usl - b.mean, b.mean - lsl) / (3 * b.sigma),
        Pp=(usl - lsl) / (6 * overall),
        Ppk=min(usl - b.mean, b.mean - lsl) / (3 * overall),
        mean_nm=b.mean,
        sigma_within_nm=b.sigma,
        phase1_signals=b.phase1_signals,
        lag1_subgroup_correlation=b.lag1,
        interpretation="Descriptive only; screening is not proof of stability or normality.",
    )

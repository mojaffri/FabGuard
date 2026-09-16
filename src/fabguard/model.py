"""Lumped ion/radical-inspired response model, with deliberately chosen coefficients."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Recipe:
    power_w: float = 750.0
    pressure_mtorr: float = 35.0
    flow_sccm: float = 50.0
    temperature_c: float = 40.0
    time_s: float = 120.0

    def __post_init__(self):
        for name, lo, hi in [
            ("power_w", 550, 950),
            ("pressure_mtorr", 20, 50),
            ("flow_sccm", 30, 75),
            ("temperature_c", 20, 60),
            ("time_s", 90, 160),
        ]:
            if not np.isfinite(getattr(self, name)) or not lo <= getattr(self, name) <= hi:
                raise ValueError(f"{name} outside educational model domain [{lo}, {hi}]")


@dataclass(frozen=True)
class Tool:
    name: str = "ETCH-01"
    gain: float = 1.0
    pressure_bias: float = 0.0

    def __post_init__(self):
        if not self.name or not np.isfinite([self.gain, self.pressure_bias]).all():
            raise ValueError("Invalid tool")
        if not 0.8 <= self.gain <= 1.2 or abs(self.pressure_bias) > 5:
            raise ValueError("Tool calibration outside model domain")


TOOLS = (
    Tool(),
    Tool("ETCH-02", 0.974, 0.5),
    Tool("ETCH-03", 1.028, -0.4),
    Tool("ETCH-04", 0.99, 0.9),
)
FAULTS = {"none", "rf_loss", "vacuum_leak", "flow_restriction", "sensor_bias"}


def response(recipe=Recipe(), tool=Tool(), age=0.0, fault="none"):
    """Noise-free latent response. age is normalized wall condition, not a measured sensor."""
    if not np.isfinite(age) or not 0 <= age <= 3 or fault not in FAULTS:
        raise ValueError("Invalid wall condition or fault")
    power = recipe.power_w * (0.86 if fault == "rf_loss" else 1)
    pressure = recipe.pressure_mtorr + tool.pressure_bias + (8 if fault == "vacuum_leak" else 0)
    flow = recipe.flow_sccm * (0.8 if fault == "flow_restriction" else 1)
    ion = (power / 750) ** 0.55 * np.exp(-(pressure - 35) / 150)
    radical = (flow / (flow + 25)) / (50 / 75)
    residence = (pressure / (pressure + 15)) / (35 / 50)
    thermal = np.exp(650 * (1 / 313.15 - 1 / (recipe.temperature_c + 273.15)))
    rate = 250 * ion * radical * residence * thermal * tool.gain * np.exp(-0.05 * age)
    nonuniformity = (
        1.3
        + 0.003 * (pressure - 33) ** 2
        + 0.000025 * (power - 730) ** 2
        + 0.001 * (flow - 52) ** 2
        + 0.9 * age
    )
    cd = 0.018 * (power - 750) + 0.1 * (pressure - 35) + 2.2 * age
    defects = float(
        1
        / (
            1
            + np.exp(-(-5 + 1.7 * age + 0.5 * (nonuniformity - 1.5) + 0.003 * max(power - 750, 0)))
        )
    )
    wear = (power / 750) ** 2 * recipe.time_s / 120 * (1 + 0.2 * radical)
    return dict(
        rate_nm_min=float(rate),
        depth_nm=float(rate * recipe.time_s / 60),
        nonuniformity_pct=float(nonuniformity),
        cd_error_nm=float(cd),
        defect_probability=defects,
        wear_index=float(wear),
        cycle_s=recipe.time_s + 45,
        pressure_mtorr=pressure,
        reflected_power_w=8 + 30 * age + (65 if fault == "rf_loss" else 0),
        flow_sccm=flow,
        endpoint_s=500 / rate * 60,
        vibration_mm_s=0.5 + 1.2 * age,
    )


def wafer(recipe=Recipe(), tool=Tool(), age=0.0, fault="none", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    truth = response(recipe, tool, age, fault)
    depth = truth["depth_nm"] + rng.normal(0, 3.5)
    uniformity = max(0, truth["nonuniformity_pct"] + rng.normal(0, 0.12))
    cd = truth["cd_error_nm"] + rng.normal(0, 0.6)
    particle = bool(rng.random() < truth["defect_probability"])
    return dict(
        depth_nm=float(depth),
        nonuniformity_pct=float(uniformity),
        cd_error_nm=float(cd),
        particle_defect=particle,
        good=bool(480 <= depth <= 520 and uniformity <= 3 and abs(cd) <= 5 and not particle),
        pressure_mtorr=float(
            truth["pressure_mtorr"] + rng.normal(0, 0.15) + (6 if fault == "sensor_bias" else 0)
        ),
        reflected_power_w=float(truth["reflected_power_w"] + rng.normal(0, 3)),
        flow_sccm=float(truth["flow_sccm"] + rng.normal(0, 0.3)),
        endpoint_s=float(truth["endpoint_s"] + rng.normal(0, 0.9)),
        vibration_mm_s=float(truth["vibration_mm_s"] + rng.normal(0, 0.12)),
        cycle_s=truth["cycle_s"],
        wear_index=truth["wear_index"],
    )

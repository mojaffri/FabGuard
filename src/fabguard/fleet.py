"""Campaign simulation with separate wafer, service-event and sensor tables."""

import numpy as np
import pandas as pd

from .model import TOOLS, Recipe, response, wafer


def simulate(seed=7, lots=240, tools=TOOLS, clean_every=70, fault="none", onset=80):
    if not isinstance(lots, int) or lots < 10 or clean_every < 1:
        raise ValueError("Need >=10 lots and positive cleaning interval")
    # Independent streams stop one tool's maintenance from changing another tool's noise.
    streams = np.random.SeedSequence(seed).spawn(len(tools))
    wafers, sensors, events = [], [], []
    for tool, stream in zip(tools, streams):
        rng = np.random.default_rng(stream)
        age, clock_s, since_clean, episode = 0.0, 0.0, 0, 0
        for lot in range(lots):
            active_fault = fault if lot >= onset else "none"
            rows = [wafer(Recipe(), tool, age, active_fault, rng) for _ in range(5)]
            begin = clock_s
            clock_s += sum(r["cycle_s"] for r in rows)
            for j, row in enumerate(rows):
                wafers.append(
                    dict(
                        campaign=seed,
                        tool=tool.name,
                        lot=lot,
                        wafer=j,
                        **row,
                        truth_age=age,
                        truth_fault=active_fault,
                    )
                )
            sensor = {
                key: float(np.mean([r[key] for r in rows]))
                for key in [
                    "pressure_mtorr",
                    "reflected_power_w",
                    "flow_sccm",
                    "endpoint_s",
                    "vibration_mm_s",
                ]
            }
            age += 0.016 * response(age=age)["wear_index"] * rng.lognormal(-0.5 * 0.15**2, 0.15)
            hazard = 0.002 + 0.28 / (1 + np.exp(-(age - 1.0) * 9))
            failure = bool(rng.random() < hazard)
            sensors.append(
                dict(
                    campaign=seed,
                    tool=tool.name,
                    lot=lot,
                    episode=episode,
                    lots_since_clean=since_clean,
                    **sensor,
                    failure_after_lot=failure,
                )
            )
            events.append(
                dict(
                    campaign=seed,
                    tool=tool.name,
                    lot=lot,
                    kind="production",
                    start_s=begin,
                    end_s=clock_s,
                    duration_s=clock_s - begin,
                    units=5,
                    good_units=sum(r["good"] for r in rows),
                )
            )
            since_clean += 1
            kind = "repair" if failure else "clean" if since_clean >= clean_every else None
            if kind:
                duration = float(rng.lognormal(np.log(2.0 * 3600), 0.3)) if failure else 1800.0
                events.append(
                    dict(
                        campaign=seed,
                        tool=tool.name,
                        lot=lot,
                        kind=kind,
                        start_s=clock_s,
                        end_s=clock_s + duration,
                        duration_s=duration,
                        units=0,
                        good_units=0,
                    )
                )
                clock_s += duration
                age, since_clean, episode = 0.0, 0, episode + 1
    return pd.DataFrame(wafers), pd.DataFrame(sensors), pd.DataFrame(events)


def reliability(events, ideal_cycle_s=135.0):
    if events.empty or not np.isfinite(ideal_cycle_s) or ideal_cycle_s <= 0:
        raise ValueError("Need events and a positive ideal cycle")
    rows = []
    for tool, data in events.groupby("tool", sort=True):
        if (data.duration_s <= 0).any() or not np.isfinite(data.duration_s).all():
            raise ValueError("Event durations must be positive and finite")
        production = data.loc[data.kind == "production"]
        uptime = float(production.duration_s.sum())
        repair = data.loc[data.kind == "repair"]
        repair_s = float(repair.duration_s.sum())
        planned = float(data.duration_s.sum())
        units, good = int(data.units.sum()), int(data.good_units.sum())
        if uptime <= 0 or units <= 0 or not 0 <= good <= units:
            raise ValueError("Invalid production accounting")
        performance = ideal_cycle_s * units / uptime
        if performance > 1 + 1e-9:
            raise ValueError("Ideal cycle exceeds demonstrated actual cycle")
        availability, quality = uptime / planned, good / units
        rows.append(
            dict(
                tool=tool,
                failures=len(repair),
                units=units,
                good_units=good,
                operating_h=uptime / 3600,
                repair_h=repair_s / 3600,
                planned_h=planned / 3600,
                MTBF_h=uptime / 3600 / len(repair) if len(repair) else None,
                MTTR_h=repair_s / 3600 / len(repair) if len(repair) else None,
                availability=availability,
                performance=performance,
                quality=quality,
                OEE=availability * performance * quality,
            )
        )
    return pd.DataFrame(rows)

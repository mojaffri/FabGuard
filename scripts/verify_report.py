"""Recompute published metrics and verify source/data provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from fabguard.fleet import reliability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("reports/reference"))
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parents[1], args.directory
    manifest = json.loads((out / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    for relative, expected in manifest["data_sha256"].items():
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == expected, relative
    metrics = json.loads((out / "metrics.json").read_text())
    events = pd.read_csv(out / "events.csv")
    measured = reliability(events)
    saved = pd.read_csv(out / "reliability.csv")
    np.testing.assert_allclose(measured.select_dtypes("number"), saved.select_dtypes("number"))
    wafers = pd.read_csv(out / "wafers.csv")
    assert len(wafers) == metrics["wafers"] == events.units.sum()
    assert wafers.good.sum() == events.good_units.sum()
    predictions = pd.read_csv(out / "maintenance_predictions.csv")
    for name in ["sensor_forest", "age_only"]:
        m = metrics["maintenance"][name]
        assert np.isclose(
            average_precision_score(predictions.failure_next_5, predictions[name]), m["PR_AUC"]
        )
        assert np.isclose(
            brier_score_loss(predictions.failure_next_5, predictions[name]), m["Brier"]
        )
    splits = metrics["maintenance"]["split_campaigns"]
    assert len(set(sum(splits.values(), []))) == sum(map(len, splits.values()))
    matching = pd.read_csv(out / "matching_confirmation.csv")
    for group in ["before", "after"]:
        means = matching.loc[matching.group == group].groupby("tool").depth_nm.mean()
        assert np.isclose(means.max() - means.min(), metrics[f"matching_{group}_span_nm"])
    history = pd.read_csv(out / "optimization_history.csv")
    for _, block in history.groupby(["seed", "method"]):
        np.testing.assert_allclose(block.best_loss, block.loss.cummin())
    print(
        "PASS: source/data hashes, event accounting, OEE, held-out predictions, split separation, matching and search histories"
    )


if __name__ == "__main__":
    main()

"""Forecast corrective failure within five future lots, before the next service."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "lots_since_clean",
    "pressure_mtorr",
    "reflected_power_w",
    "flow_sccm",
    "endpoint_s",
    "vibration_mm_s",
]


def labels(sensors):
    horizon = 5
    records = []
    for _, block in sensors.groupby(["campaign", "tool", "episode"]):
        block = block.sort_values("lot")
        rows = block.to_dict("records")
        for i, row in enumerate(rows):
            # At decision time the current lot is complete; already-failed rows are not scored.
            if row["failure_after_lot"]:
                continue
            future = rows[i + 1 : i + 1 + horizon]
            positive = any(r["failure_after_lot"] for r in future)
            # An early repair confirms a positive; clean/end-of-campaign truncation is censored.
            if len(future) == horizon or positive:
                records.append(dict(**row, failure_next_5=int(positive)))
    return pd.DataFrame(records)


def scores(y, probability, threshold):
    return dict(
        PR_AUC=float(average_precision_score(y, probability)),
        Brier=float(brier_score_loss(y, probability)),
        precision=float(precision_score(y, probability >= threshold, zero_division=0)),
        recall=float(recall_score(y, probability >= threshold, zero_division=0)),
        prevalence=float(np.mean(y)),
        n=len(y),
        threshold=float(threshold),
    )


def evaluate(train, validation, test, seed=7):
    sets = [set(d.campaign) for d in (train, validation, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Campaigns must be disjoint across train, validation and test")
    if any(d.failure_next_5.nunique() != 2 for d in (train, validation, test)):
        raise ValueError("Each split needs positive and negative cases")
    models = {
        "sensor_forest": (
            RandomForestClassifier(
                n_estimators=160, min_samples_leaf=20, max_depth=7, random_state=seed, n_jobs=1
            ),
            FEATURES,
        ),
        "age_only": (make_pipeline(StandardScaler(), LogisticRegression()), ["lots_since_clean"]),
    }
    metrics, predictions = {}, test[["campaign", "tool", "lot", "failure_next_5"]].copy()
    for name, (model, columns) in models.items():
        model.fit(train[columns], train.failure_next_5)
        val_p = model.predict_proba(validation[columns])[:, 1]
        candidates = np.linspace(0.05, 0.8, 31)
        # F1 selected on validation campaigns only; no optimization against test metrics.
        f1 = []
        for threshold in candidates:
            p = precision_score(validation.failure_next_5, val_p >= threshold, zero_division=0)
            r = recall_score(validation.failure_next_5, val_p >= threshold, zero_division=0)
            f1.append(2 * p * r / (p + r) if p + r else 0)
        threshold = float(candidates[int(np.argmax(f1))])
        probability = model.predict_proba(test[columns])[:, 1]
        predictions[name] = probability
        metrics[name] = scores(test.failure_next_5, probability, threshold)
    constant = np.full(len(test), train.failure_next_5.mean())
    metrics["constant_prevalence"] = scores(test.failure_next_5, constant, 0.5)
    # Resample independent campaigns, not overlapping five-lot labels.
    rng, boot = np.random.default_rng(seed + 90), []
    campaigns = test.campaign.unique()
    for _ in range(400):
        sample = pd.concat(
            [
                predictions.loc[predictions.campaign == c]
                for c in rng.choice(campaigns, len(campaigns), replace=True)
            ]
        )
        boot.append(average_precision_score(sample.failure_next_5, sample.sensor_forest))
    metrics["sensor_forest"]["PR_AUC_campaign_bootstrap_95"] = np.quantile(
        boot, [0.025, 0.975]
    ).tolist()
    metrics["split_campaigns"] = {
        k: sorted(map(int, v)) for k, v in zip(["train", "validation", "test"], sets)
    }
    metrics["features"] = FEATURES
    return predictions, metrics

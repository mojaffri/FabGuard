# FabGuard reference experiment

All results are synthetic. No real fab data or physical experiments.

Seed 7; 4,800 fleet wafers; 30 independent SPC trials per scenario.

![Equipment analytics](overview.png)

## Drift detection

Delays count lots after onset (zero means the first affected lot). Undetected trials remain censored; false alarms are retained.

| Chart | Median detected delay / lots | Detected trials | Stable point alarm fraction |
|---|---:|---:|---:|
| xbar | 11.0 | 30/30 | 0.633% |
| r | 23.0 | 11/30 | 0.800% |
| ewma | 8.0 | 30/30 | 1.233% |
| cusum | 8.0 | 30/30 | 4.267% |

CUSUM is not reset after an alarm; point alarm fractions include persistent excursions. These are not average-run-length estimates or familywise error rates. The wall challenge changes the mean, not within-group dispersion: R-chart flags here are incidental false alarms, not evidence of sensitivity to wall drift. Delay comparisons are not adjusted to equal false-alarm rates.

![SPC](spc.png)

## Tool matching

Across-tool mean depth span: 27.71 → 1.76 nm on independent confirmation groups (60 wafers per tool and condition).
Offsets are estimated only from calibration wafers. Clean-tool time compensation does not fix uniformity, sensor faults, or aging. See matching.csv for Welch intervals and ±3 nm equivalence checks.

## Maintenance forecast

| Model | Test PR-AUC | Brier score | Precision | Recall |
|---|---:|---:|---:|---:|
| sensor_forest | 0.440 | 0.071 | 0.420 | 0.623 |
| age_only | 0.451 | 0.070 | 0.374 | 0.678 |
| constant_prevalence | 0.104 | 0.093 | 0.000 | 0.000 |

Test prevalence: 10.4%; campaign-bootstrap PR-AUC interval: [0.4119342601767906, 0.4976539324611727]. Only four test campaigns; uncertainty estimates are coarse.
The label excludes current failures and censored futures. No latent wall condition or future event is a feature. Forecast accuracy is not evidence of maintenance cost savings; no learned maintenance policy is deployed.

## Recipe search

| Method | Mean loss ↓ | Confirmation good fraction | Wafers/h | Wear index ↓ |
|---|---:|---:|---:|---:|
| bayesian | 2.311 | 86.4% | 21.15 | 1.075 |
| nominal | 2.890 | 72.0% | 21.82 | 1.200 |
| random | 2.312 | 85.7% | 20.88 | 1.142 |

Three search seeds; equal evaluation budget and initial design. Confirmation uses new wafer noise and wall conditions. The loss is a chosen weighted objective with soft penalties, not a validated cost model or a guarantee of feasibility.

## Reliability accounting

| Tool | Failures | MTBF / h | MTTR / h | Availability | Quality | OEE |
|---|---:|---:|---:|---:|---:|---:|
| ETCH-01 | 4 | 13.75 | 1.94 | 87.7% | 80.2% | 57.6% |
| ETCH-02 | 5 | 11.00 | 1.79 | 86.0% | 39.1% | 27.5% |
| ETCH-03 | 6 | 9.17 | 2.30 | 80.0% | 97.3% | 63.7% |
| ETCH-04 | 5 | 11.00 | 1.43 | 88.5% | 67.9% | 49.2% |
MTBF is observed operating exposure / corrective failures. MTTR is completed repair time / repairs. Scheduled cleans count against availability but not corrective MTTR. All event durations, including final service, are inside the recorded campaign horizon. OEE = availability × performance × good fraction; the ideal cycle is 135 s/wafer.

Source/dependency manifest, raw CSVs, frozen baselines, predictions and optimizer histories accompany this report.

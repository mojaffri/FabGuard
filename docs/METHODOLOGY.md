# Modeling and analysis decisions

## Etch response and units

Nominal inputs are 750 W RF power, 35 mTorr pressure, 50 sccm reactive-gas flow, 40 °C wafer temperature and 120 s etch duration. These are illustrative settings, not an operating procedure. Temperature is converted to kelvin in the thermal factor. Gas identity, substrate and film chemistry are unspecified; consequently this is not a quantitative prediction for a particular material.

The model multiplies normalized factors:

```text
I = (P_RF / 750)^0.55 exp(-(pressure - 35)/150)
F = [flow/(flow+25)] / (50/75)
R = [pressure/(pressure+15)] / (35/50)
T = exp[650(1/313.15 - 1/(temperature_C+273.15))]
etch_rate_nm/min = 250 I F R T tool_gain exp(-0.05 wall_condition)
depth_nm = etch_rate_nm/min × etch_time_s / 60
```

The RF factor represents increasing ion assistance; saturating flow represents finite reactive-species supply; pressure balances residence and a simplified collision penalty. Their forms encode qualitative trends, not a fitted mechanistic model. All coefficients, bounds and nominal values were selected for this educational experiment. Nominal depth is 500 nm.

Nonuniformity is a positive quadratic summary in pressure, RF and flow plus wall condition. CD error is a linear RF/pressure response plus degradation. Particle probability is logistic in wall condition, nonuniformity and high RF. Wear scales with RF squared and exposure time. Code in `model.py` is the complete coefficient specification.

Measured depth adds independent Gaussian noise with 3.5 nm SD, combining wafer variation and metrology noise. Nonuniformity and CD use separate noise terms. Particle outcomes are Bernoulli draws. A wafer is good only if depth is within [480,520] nm, nonuniformity ≤3%, absolute CD error ≤5 nm and no particle defect is drawn. No assumption that these illustrative specifications match a real NXP process is made.

## Tool state and failures

Four tool gains are 1.000, 0.974, 1.028 and 0.990. Small pressure calibration offsets also differ. Wall condition increases after each five-wafer lot by 0.016 times nominal-recipe wear and lognormal variation. Reflected RF and vibration increase with this hidden state; endpoint time follows etch rate. Sensors contain noise.

The corrective-failure probability per completed lot is `0.002 + 0.28 sigmoid(9(age−1))`. Repair time is lognormal with 2 h median and 0.3 log SD. Repair resets condition; otherwise a scheduled clean after 70 lots takes 0.5 h and also resets condition. Failure is evaluated after the lot, so this simplified model does not simulate mid-wafer aborts or extra scrapped work in process.

The independent SPC challenge injects RF power loss (14%), a physical pressure increase (8 mTorr), a flow reduction (20%), measured pressure bias (6 mTorr), or gradual wall drift. The sensor bias leaves the physical response unchanged because this model does not use feedback to control pressure. That deliberate negative-control scenario demonstrates a blind spot in depth-only SPC.

## SPC and capability

Each rational subgroup is five consecutive same-recipe wafers on one tool with fixed wall condition during the group. A separate 40-group baseline estimates center, average range and within-group SD `Rbar/2.326`. NIST constants for n=5 are A2=0.577, D3=0 and D4=2.114. X-bar limits are center ± A2 Rbar; range limits are [D3 Rbar, D4 Rbar].

EWMA starts at the baseline center, uses λ=0.2 and finite-time three-sigma limits based on subgroup standard error `sigma/sqrt(5)`. The two one-sided CUSUMs operate on standardized subgroup means, using k=0.5 and h=5. CUSUM is not reset after an alarm; repeated point alarms therefore do not represent independent failure events.

Limits remain frozen throughout Phase II. Baseline signals and lag-one subgroup correlation are retained. Cp and Cpk use within-group SD; Pp and Ppk use overall SD. No out-of-control subgroup is silently deleted. These are descriptive indices, not accepted production capability claims; normality, independence, stability and measurement adequacy require further assessment.

Repeated SPC experiments report detection delay conditional on detection, detected count, and stable point alarm fractions. Missed detections are missing values in CSV, not zero. R is primarily a dispersion chart, so a location-only drift can remain invisible to it. For the stable and sensor-bias negative controls, post-onset flags are false alarms rather than successful fault detections. The charts have not been tuned to equal average run length; comparing their delays alone does not prove one is universally better.

## Tool-to-tool matching

At clean state and nominal recipe, collect 60 calibration wafers per tool. Recommend `time_new = 120 × mean_reference / mean_tool`, bounded to the time domain. Time proportionality follows the local model; real tools would require a new response study. Confirm before and after using separate independent wafers. Welch intervals quantify each tool's mean difference from the reference. The full 95% interval must lie within ±3 nm to label equivalence. Intervals are per comparison and not familywise-adjusted.

This is mean matching at one wall condition. It does not establish uniformity equivalence, fleet-wide capability or continued matching after chamber aging. Across-tool mean span is max tool mean minus min tool mean, not an individual-wafer error metric.

## Reliability and OEE

MTBF = accumulated production operating hours / corrective failures. MTTR = accumulated completed corrective repair hours / repairs. With no failures these estimates are undefined. Production time includes etching plus 45 s handling per wafer. Scheduled cleans count in the planned production horizon and reduce availability but are excluded from corrective MTTR.

Availability = production time / total recorded time. Performance = ideal cycle (135 s) × total wafers / production time. Quality = good wafers / total wafers. OEE is their product. The ideal cycle corresponds to the shortest modeled etch duration (90 s) plus handling. Campaigns end after any final service event, so all recorded service is fully observed; this is an explicit study-horizon convention. Summed tool-hours are exposure, not elapsed fleet wall-clock time.

## Maintenance evaluation

At the end of a non-failed lot, predict whether corrective failure will occur in the next five lots before any service. Exclude failed current rows. Early repair confirms a positive; shorter negative futures at a scheduled clean or end of campaign are censored and excluded. This estimates risk conditional on observed service policy and follow-up availability; censoring is informative and can bias comparisons to deployments with different policies.

Train on eight campaigns, select thresholds on two different campaigns, evaluate on four untouched campaigns. Features: lots since clean, pressure, reflected RF, flow, endpoint time and vibration. Campaign ID, lot ID, episode ID, future failure and hidden age are excluded. Forest hyperparameters are fixed. Compare against age-only logistic regression and constant training prevalence. Each model's F1 threshold is selected on validation data only. Report PR-AUC, Brier, precision and recall. A 400-resample campaign bootstrap gives a coarse PR-AUC interval; overlapping windows are not resampled as independent points.

No policy acts on these predictions. Demonstrating actual maintenance value requires a prospective intervention simulation including cleaning cost, lost production and false alarms. A good classifier score alone does not establish savings.

## Bayesian recipe optimization

Search RF, pressure, flow and time; hold temperature at 40 °C. Normalize inputs to [0,1]^4. Use a 10-point Latin-hypercube initial design with one point replaced by the nominal recipe. Both methods share it. GP regression uses unit-amplitude Matérn-5/2, fixed normalized length scale 0.35, normalized output, and 1e−6 diagonal regularization. Expected improvement for minimization uses an exploration offset of 0.01 and selects from 768 uniformly sampled candidates each iteration. This is a discrete candidate approximation to acquisition maximization.

Evaluate each recipe across four tools and wall conditions 0, 0.4 and 0.8, averaging:

```text
2(depth_error/20)^2 + 0.4(nonuniformity/3)^2
+ 0.8(defect_probability/0.05) + 0.6(cycle_time/165)
+ 0.3(wear/1.2) + 8(specification_violation)^2
```

Violation sums depth exceedance beyond ±20 nm divided by 20, nonuniformity exceedance beyond 3%, and absolute CD exceedance beyond 5 nm divided by 5. Weights and normalization are assumptions, not measured business costs. Soft penalties do not enforce hard feasibility. The report therefore includes observed confirmation good fraction, depth RMSE, particle fraction, throughput and wear alongside loss.

Each search gets 30 evaluations and three independent seeds. The random comparator shares the initial design, then samples uniformly. New confirmation wafers use independent noise and wall condition uniformly drawn from [0,0.9]. The nominal recipe is confirmed as a third comparator. Results transfer only within the simulator family; a different kinetic law or actual tool can reverse the ranking.

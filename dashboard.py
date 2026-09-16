"""Local dashboard. Every displayed result is from simulation."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fabguard.model import TOOLS, Recipe, response

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="FabGuard | Etch equipment analytics", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f4f6f8} .block-container {padding-top:4rem;max-width:1500px}
h1,h2,h3 {color:#163443} [data-testid="stMetric"] {background:white;border:1px solid #dde5e9;padding:16px;border-radius:8px}
[data-testid="stMetricValue"] {color:#126b78} .eyebrow {font-size:12px;letter-spacing:2px;color:#126b78;font-weight:700}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="eyebrow">MANUFACTURING ANALYTICS / ENGINEERING LAB</div>', unsafe_allow_html=True
)
st.title("FabGuard")
st.write("Plasma-etch equipment · process stability · maintenance decisions")
st.caption(
    "SIMULATION ONLY — Synthetic wafers and sensor signals. Uncalibrated educational model; no connection to physical equipment."
)

with st.sidebar:
    st.subheader("Evidence workspace")
    folder = st.selectbox("Experiment", ["reports/reference", "runs/latest"])
    st.caption("Reproduce a new experiment with the README command, then select runs/latest.")
    section = st.radio(
        "Analysis",
        [
            "Process monitoring",
            "Tool matching",
            "Reliability",
            "Maintenance forecast",
            "Recipe optimization",
            "Recipe sandbox",
        ],
    )
    st.divider()
    st.write("**Engineering question**")
    questions = {
        "Process monitoring": "Is this process changing, even while measurements remain inside specification?",
        "Tool matching": "Can calibration reduce between-tool mean offsets on new wafers?",
        "Reliability": "Where do availability, speed and quality losses enter OEE?",
        "Maintenance forecast": "Do measured sensors add information beyond time since cleaning?",
        "Recipe optimization": "What quality, throughput and wear tradeoffs survive independent confirmation?",
        "Recipe sandbox": "How do recipe and wall-condition assumptions change the modeled response?",
    }
    st.caption(questions[section])

directory = ROOT / folder
if not (directory / "metrics.json").exists():
    st.info("Generate this experiment first: python -m fabguard.experiments --out runs/latest")
    st.stop()


@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


def table(name):
    path = directory / f"{name}.csv"
    return read_csv(str(path), path.stat().st_mtime_ns)


metrics = json.loads((directory / "metrics.json").read_text())
st.caption(
    f"Experiment seed {metrics['seed']} · {'Quick validation run' if metrics['quick'] else 'Full reference run'} · {metrics['wafers']:,} fleet wafers"
)

if section == "Process monitoring":
    st.subheader("Detect a change. Investigate its cause.")
    charts = table("charts")
    scenario = st.selectbox("Injected scenario", charts.scenario.unique(), index=1)
    c = charts.loc[charts.scenario == scenario]
    caps = metrics["capability"]
    a, b, d = st.columns(3)
    a.metric("Baseline Cp", f"{caps['Cp']:.2f}")
    b.metric("Baseline Cpk", f"{caps['Cpk']:.2f}")
    d.metric("Phase-I flagged subgroups", caps["phase1_signals"])
    st.caption(
        "40 baseline groups × 5 consecutive wafers on ETCH-01. Limits are frozen. Capability is descriptive; these diagnostics alone do not qualify the process."
    )
    choice = st.selectbox("Control chart", ["X-bar", "Range", "EWMA", "CUSUM"])
    fig = go.Figure()
    if choice == "CUSUM":
        for key in ["cusum_plus", "cusum_minus"]:
            fig.add_trace(go.Scatter(x=c.lot, y=c[key], name=key))
        fig.add_hline(y=5, line_dash="dash", annotation_text="Decision threshold")
        ylabel = "Standardized cumulative deviation"
    else:
        value, lower, upper = {
            "X-bar": ("xbar", "xbar_lcl", "xbar_ucl"),
            "Range": ("range", "r_lcl", "r_ucl"),
            "EWMA": ("ewma", "ewma_lcl", "ewma_ucl"),
        }[choice]
        fig.add_trace(go.Scatter(x=c.lot, y=c[value], name=choice, line_color="#126b78"))
        for key in [lower, upper]:
            fig.add_trace(
                go.Scatter(x=c.lot, y=c[key], name=key, line=dict(dash="dash", color="#b65333"))
            )
        ylabel = "Depth / nm" if choice != "Range" else "Within-subgroup range / nm"
    if scenario != "stable":
        fig.add_vline(x=30, line_dash="dot", annotation_text="Injection begins")
    fig.update_layout(
        height=410,
        xaxis_title="Phase-II lot (5 wafers)",
        yaxis_title=ylabel,
        legend_orientation="h",
    )
    st.plotly_chart(fig, width="stretch")
    if scenario == "sensor_bias":
        st.info(
            "A pressure-sensor offset does not change physical depth in this open-loop model. Depth SPC can miss the fault; sensor redundancy or consistency checks would be needed."
        )
    trials = table("detection_trials")
    st.write("**Repeated trials — delays and retained false alarms**")
    st.dataframe(trials.loc[trials.scenario == scenario], hide_index=True)
    st.download_button(
        "Download plotted chart data", c.to_csv(index=False), "fabguard-chart.csv", "text/csv"
    )

elif section == "Tool matching":
    st.subheader("Calibrate once. Confirm on new wafers.")
    a, b = st.columns(2)
    a.metric("Across-tool mean span · before", f"{metrics['matching_before_span_nm']:.2f} nm")
    b.metric("Across-tool mean span · after", f"{metrics['matching_after_span_nm']:.2f} nm")
    data = table("matching_confirmation")
    st.plotly_chart(
        px.box(
            data,
            x="tool",
            y="depth_nm",
            color="group",
            points=False,
            color_discrete_sequence=["#b65333", "#126b78"],
        ),
        width="stretch",
    )
    st.dataframe(table("matching"), hide_index=True)
    st.caption(
        "60 calibration wafers per tool; 60 independent confirmation wafers per tool and condition. Proposed etch-time corrections match ETCH-01. Equivalence requires the entire 95% Welch interval inside ±3 nm. This study holds wall condition clean."
    )

elif section == "Reliability":
    st.subheader("Account for every hour and wafer.")
    data = table("reliability")
    st.plotly_chart(
        px.bar(
            data.melt(id_vars="tool", value_vars=["availability", "performance", "quality", "OEE"]),
            x="tool",
            y="value",
            color="variable",
            barmode="group",
        ),
        width="stretch",
    )
    st.dataframe(data, hide_index=True)
    st.caption(
        "MTBF = operating hours / failures; MTTR = repair hours / repairs. Cleans and repairs reduce availability. Performance uses a fixed 135 s ideal cycle. Quality counts wafers passing depth, uniformity, CD and particle criteria."
    )
    tool = st.selectbox("Tool event log", data.tool)
    events = table("events")
    selected = events.loc[events.tool == tool]
    st.dataframe(selected, hide_index=True, height=270)
    st.download_button(
        "Download event log", selected.to_csv(index=False), "fabguard-events.csv", "text/csv"
    )

elif section == "Maintenance forecast":
    st.subheader("Predict a corrective failure in the next five lots.")
    m = metrics["maintenance"]
    cols = st.columns(3)
    for col, name in zip(cols, ["sensor_forest", "age_only", "constant_prevalence"]):
        col.metric(name.replace("_", " ").title() + " · PR-AUC", f"{m[name]['PR_AUC']:.3f}")
    st.caption(
        f"Test prevalence {m['sensor_forest']['prevalence']:.1%}. Thresholds chosen only on validation campaigns. Current failures and incomplete future windows excluded."
    )
    st.dataframe(
        pd.DataFrame({k: m[k] for k in ["sensor_forest", "age_only", "constant_prevalence"]}).T
    )
    predictions = table("maintenance_predictions")
    campaign = st.selectbox("Held-out campaign", predictions.campaign.unique())
    tool = st.selectbox("Chamber", predictions.tool.unique())
    selected = predictions.loc[(predictions.campaign == campaign) & (predictions.tool == tool)]
    st.plotly_chart(
        px.line(
            selected,
            x="lot",
            y=["sensor_forest", "age_only", "failure_next_5"],
            labels={"value": "Probability / binary outcome"},
        ),
        width="stretch",
    )
    st.info(
        "Sensors reflect simulated degradation. No real-world predictive accuracy or maintenance savings are established. Overlapping forecast windows are correlated; uncertainty resamples whole campaigns."
    )

elif section == "Recipe optimization":
    st.subheader("Inspect the tradeoff, not just the winning score.")
    history = table("optimization_history")
    seed = st.selectbox("Search seed", history.seed.unique())
    st.plotly_chart(
        px.line(
            history.loc[history.seed == seed],
            x="evaluation",
            y="best_loss",
            color="method",
            labels={"best_loss": "Best synthetic loss ↓"},
        ),
        width="stretch",
    )
    st.dataframe(table("optimization_confirmation"), hide_index=True)
    st.caption(
        "Gaussian process with Matérn kernel and expected improvement; equal-budget random-search comparator shares the initial design. Objective combines depth error, nonuniformity, defects, cycle time and wear, plus soft specification penalties. New stochastic wafers confirm each recommendation."
    )

else:
    st.subheader("Recipe response sandbox")
    a, b, c = st.columns(3)
    power = a.slider("RF power / W", 550, 950, 750)
    pressure = b.slider("Pressure / mTorr", 20, 50, 35)
    flow = c.slider("Reactive gas flow / sccm", 30, 75, 50)
    duration = a.slider("Etch duration / s", 90, 160, 120)
    temp = b.slider("Wafer temperature / °C", 20, 60, 40)
    age = c.slider("Normalized wall condition", 0.0, 1.5, 0.0, 0.05)
    tool_name = st.selectbox("Tool", [t.name for t in TOOLS])
    tool = next(t for t in TOOLS if t.name == tool_name)
    r = response(Recipe(power, pressure, flow, temp, duration), tool, age)
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Mean depth", "Nonuniformity", "Defect probability", "Wear index"],
        [
            f"{r['depth_nm']:.1f} nm",
            f"{r['nonuniformity_pct']:.2f}%",
            f"{r['defect_probability']:.1%}",
            f"{r['wear_index']:.2f}",
        ],
    ):
        col.metric(label, value)
    st.caption(
        "Noise-free model response. Nonuniformity is an empirical summary surrogate, not a simulated radial wafer map. Specs: depth 480–520 nm, nonuniformity ≤3%, |CD error| ≤5 nm."
    )
    st.json(r)

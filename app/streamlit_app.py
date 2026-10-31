"""UniAthleteIQ runner risk monitor.

Run `notebooks/02_leakage_free_pipeline.ipynb` first (it saves the model), then:
    streamlit run app/streamlit_app.py
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from uniathleteiq import explain, paths

st.set_page_config(page_title="UniAthleteIQ: runner risk monitor", layout="wide")

MODEL_PATH = paths.MODELS_DIR / "runner_model.joblib"
HOLDOUT_PATH = paths.MODELS_DIR / "runner_holdout_predictions.csv.gz"

def friendly(name):
    return explain.feature_label(name)


@st.cache_resource
def load():
    bundle = joblib.load(MODEL_PATH)
    hold = pd.read_csv(HOLDOUT_PATH)
    X = hold[bundle["features"]]
    contrib = explain.linear_contributions(bundle["model"], X)
    return bundle, hold, X, contrib


if not MODEL_PATH.exists() or not HOLDOUT_PATH.exists():
    st.error("Model not found. Run `notebooks/02_leakage_free_pipeline.ipynb` first.")
    st.stop()

bundle, hold, X, contrib = load()
base_rate = bundle["base_rate"]
hold["relative_risk"] = hold["risk_probability"] / base_rate
hold["flagged"] = hold["risk_score"] >= bundle["threshold"]

st.title("Runner injury-risk monitor")
m = bundle["holdout_metrics"]
st.caption(
    f"These athletes were **never seen during training**. On them the model has ROC-AUC "
    f"{m['roc_auc']:.2f}: it ranks risky days better than chance but is far from certain. "
    f"Use it to prompt a conversation with the athlete, not as a diagnosis."
)

athlete = st.sidebar.selectbox("Athlete", sorted(hold["Athlete ID"].unique()))
a = hold[hold["Athlete ID"] == athlete].sort_values("Date").reset_index()

c1, c2, c3 = st.columns(3)
c1.metric("Days recorded", len(a))
c2.metric("Injury days", int(a["injury"].sum()))
c3.metric("Days flagged high-risk", f"{a['flagged'].mean():.0%}")

line = alt.Chart(a).mark_line(strokeWidth=2, color="#2a78d6").encode(
    x=alt.X("Date:Q", title="Day"),
    y=alt.Y("relative_risk:Q", title="Risk vs average day (×)"),
    tooltip=[alt.Tooltip("Date:Q", title="Day"),
             alt.Tooltip("relative_risk:Q", title="Risk vs average (×)", format=".2f"),
             alt.Tooltip("injury:Q", title="Injured")],
)
avg = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4], color="#8a8984").encode(y="y:Q")
injuries = alt.Chart(a[a["injury"] == 1]).mark_point(
    shape="triangle-up", size=110, filled=True, color="#eb6834", stroke="white", strokeWidth=1.5
).encode(x="Date:Q", y="relative_risk:Q",
         tooltip=[alt.Tooltip("Date:Q", title="Injury on day")])
st.altair_chart((line + avg + injuries).properties(height=320), width="stretch")
st.caption("Blue line: model risk relative to an average day (dashed line = 1×). Orange triangles: actual injuries.")

st.subheader("Why is risk high or low on a given day?")
day = st.select_slider("Day", options=a["Date"].tolist(), value=int(a.loc[a["risk_score"].idxmax(), "Date"]))
row = a[a["Date"] == day].iloc[0]
idx = row["index"]

d1, d2, d3 = st.columns(3)
d1.metric("Risk vs average day", f"{row['relative_risk']:.1f}×")
d2.metric("Alert", "⚠ High risk" if row["flagged"] else "✓ Normal")
d3.metric("Injured that day", "Yes" if row["injury"] else "No")

c = contrib.loc[idx].sort_values()
reasons = pd.DataFrame({"factor": [friendly(f) for f in c.index], "contribution": c.values,
                        "value": [X.loc[idx, f] for f in c.index]})
top = pd.concat([reasons.head(5), reasons.tail(5)]).drop_duplicates("factor")
top["direction"] = np.where(top["contribution"] > 0, "raises risk", "lowers risk")
bars = alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=18).encode(
    x=alt.X("contribution:Q", title="Effect on log-odds of injury"),
    y=alt.Y("factor:N", sort="-x", title=None),
    color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                     range=["#eb6834", "#2a78d6"]), title=None),
    tooltip=["factor", alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("contribution:Q", format=".3f")],
)
st.altair_chart(bars.properties(height=360), width="stretch")

with st.expander("All feature values for this day"):
    st.dataframe(pd.DataFrame({"feature": [friendly(f) for f in X.columns],
                               "value": X.loc[idx].values}), hide_index=True)

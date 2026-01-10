import os
import pandas as pd
import streamlit as st

from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")
st.title("Model Monitoring & Feedback Dashboard")

# ---- Manual refresh (helps Streamlit Cloud) ----
if st.button("🔄 Refresh logs"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=5)
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    df = pd.read_csv(LOG_PATH)

    # Make sure required columns exist (avoid crashes)
    expected_cols = [
        "timestamp", "model_version", "latency_ms",
        "feedback_score", "feedback_text"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # Parse timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", na_position="last")

    return df

logs = load_logs()

# Handle "no logs yet"
if logs.empty:
    st.warning(
        "No monitoring logs found yet. "
        "Please run the prediction app, submit feedback at least once, and then refresh this page."
    )
    st.stop()

# ---- Filters ----
st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].dropna().unique().tolist())
selected_model = st.sidebar.selectbox("Model version", models)

filtered = logs if selected_model == "All" else logs[logs["model_version"] == selected_model]

# ---- Key metrics ----
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", int(len(filtered)))

avg_fb = filtered["feedback_score"].dropna()
col2.metric("Avg Feedback Score", f"{avg_fb.mean():.2f}" if not avg_fb.empty else "N/A")

avg_lat = filtered["latency_ms"].dropna()
col3.metric("Avg Latency (ms)", f"{avg_lat.mean():.1f}" if not avg_lat.empty else "N/A")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "💬 Feedback Analysis", "📄 Raw Logs"])

with tab1:
    st.subheader("Model Version Comparison (Aggregated)")
    summary = logs.groupby("model_version", dropna=True).agg(
        avg_feedback_score=("feedback_score", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_predictions=("model_version", "count"),
    ).reset_index()

    st.dataframe(
        summary.style.format({
            "avg_feedback_score": "{:.2f}",
            "avg_latency_ms": "{:.1f}",
        }),
        use_container_width=True
    )

with tab2:
    st.subheader("Average Feedback Score by Model Version")
    fb = logs.groupby("model_version", dropna=True)["feedback_score"].mean().reset_index()

    if fb.empty:
        st.info("No feedback scores available yet.")
    else:
        st.bar_chart(fb.set_index("model_version"))

    st.subheader("Recent Comments")
    comments = logs.copy()
    comments["feedback_text"] = comments["feedback_text"].astype(str)
    comments = comments[comments["feedback_text"].str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)

    if comments.empty:
        st.info("No qualitative comments yet.")
    else:
        for _, row in comments.iterrows():
            ts = row["timestamp"]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else "N/A"
            st.write(f"**[{ts_str}] {row['model_version']} – Score: {row['feedback_score']}**")
            st.write(row["feedback_text"])
            st.markdown("---")

with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered, use_container_width=True)


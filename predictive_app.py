# predictive_app.py
import time
import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Sales Prediction App with Monitoring", layout="centered")
st.title("Sales Prediction App (v1 vs v2) with Live Monitoring")


@st.cache_resource
def load_models():
    model_v1 = joblib.load("revenue_model_v1.pkl")  # trained on ["Row ID"]
    model_v2 = joblib.load("revenue_model_v2.pkl")  # trained on ["Row ID", "Region", "Sub-Category"]
    return model_v1, model_v2


@st.cache_data
def load_reference_categories(csv_path="sales.csv"):
    """
    Optional: load unique categories from the dataset so dropdowns match real values.
    If sales.csv is missing, fall back to a default list.
    """
    try:
        df = pd.read_csv(csv_path)
        regions = sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else []
        subcats = sorted(df["Sub-Category"].dropna().unique().tolist()) if "Sub-Category" in df.columns else []
        return regions, subcats
    except Exception:
        return [], []


model_v1, model_v2 = load_models()
regions_list, subcats_list = load_reference_categories()

# ---------- Initialise session state ----------
defaults = {
    "pred_ready": False,
    "v1_pred": None,
    "v2_pred": None,
    "v1_latency_ms": None,
    "v2_latency_ms": None,
    "input_summary": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- INPUT SECTION ----------
st.sidebar.header("Input Parameters")

row_id = st.sidebar.number_input("Row ID", min_value=1, value=1, step=1)

# If we could load real categories from sales.csv, use them; otherwise fall back.
if regions_list:
    region = st.sidebar.selectbox("Region", regions_list)
else:
    region = st.sidebar.selectbox("Region", ["Central", "East", "South", "West"])

if subcats_list:
    subcat = st.sidebar.selectbox("Sub-Category", subcats_list)
else:
    subcat = st.sidebar.selectbox("Sub-Category", ["Chairs", "Phones", "Binders", "Storage"])

# Canonical input dataframe (for display)
input_df = pd.DataFrame({
    "Row ID": [int(row_id)],
    "Region": [region],
    "Sub-Category": [subcat],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    # v1: baseline – only uses Row ID (measure latency separately)
    t0 = time.time()
    input_v1 = input_df[["Row ID"]]
    v1_pred = model_v1.predict(input_v1)[0]
    v1_latency_ms = (time.time() - t0) * 1000.0

    # v2: improved – uses Row ID + Region + Sub-Category (measure latency separately)
    t1 = time.time()
    input_v2 = input_df[["Row ID", "Region", "Sub-Category"]]
    v2_pred = model_v2.predict(input_v2)[0]
    v2_latency_ms = (time.time() - t1) * 1000.0

    # Store in session_state so they survive reruns
    st.session_state["v1_pred"] = float(v1_pred)
    st.session_state["v2_pred"] = float(v2_pred)
    st.session_state["v1_latency_ms"] = float(v1_latency_ms)
    st.session_state["v2_latency_ms"] = float(v2_latency_ms)
    st.session_state["input_summary"] = f"Row ID={int(row_id)}, Region={region}, Sub-Category={subcat}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS IF READY ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions (Sales)")
    st.write(f"Model v1 (baseline - Row ID only): **${st.session_state['v1_pred']:,.2f}**")
    st.write(f"Latency v1: {st.session_state['v1_latency_ms']:.1f} ms")
    st.write("---")
    st.write(f"Model v2 (improved - Row ID + Region + Sub-Category): **${st.session_state['v2_pred']:,.2f}**")
    st.write(f"Latency v2: {st.session_state['v2_latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)
feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models (now each has its own latency)
        log_prediction(
            model_version="v1",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["v1_pred"],
            latency_ms=st.session_state["v1_latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        log_prediction(
            model_version="v2",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["v2_pred"],
            latency_ms=st.session_state["v2_latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )

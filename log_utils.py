# log_utils.py
import os
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "monitoring_logs.csv")

def log_prediction(
    model_version,
    model_type,
    input_summary,
    prediction,
    latency_ms,
    feedback_score,
    feedback_text,
):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "model_type": model_type,
        "input_summary": input_summary,
        "prediction": float(prediction),
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "feedback_score": int(feedback_score) if feedback_score is not None else None,
        "feedback_text": feedback_text or "",
    }

    df = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)

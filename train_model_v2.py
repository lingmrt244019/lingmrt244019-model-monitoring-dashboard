import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load dataset
df = pd.read_csv("sales.csv")

# Improved feature set
X = df[["Row ID", "Region", "Sub-Category"]]
y = df["Sales"]

# Preprocess categorical features
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["Region", "Sub-Category"])
    ],
    remainder="passthrough"
)

# Build pipeline
model_v2 = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# Train model
model_v2.fit(X, y)

# Save model
joblib.dump(model_v2, "revenue_model_v2.pkl")
print("Improved model v2 saved.")

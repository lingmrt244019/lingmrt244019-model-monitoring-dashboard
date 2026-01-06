import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("sales.csv")

# Baseline model: numeric proxy only
X = df[["Row ID"]]
y = df["Sales"]

model_v1 = LinearRegression()
model_v1.fit(X, y)

joblib.dump(model_v1, "revenue_model_v1.pkl")
print("Baseline model v1 saved.")

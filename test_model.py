import pandas as pd
import joblib
from train_model import feature_engineering
from sklearn.metrics import r2_score

# --- Load Data ---
test_data = pd.read_csv("test.csv")

# --- Apply Feature Engineering ---
test_data = feature_engineering(test_data)

# --- Load Model ---
model = joblib.load("ridge_pipeline.pkl")   

# --- Separate features and target ---

y_test = test_data["trip_duration"]
X_test = test_data.drop(columns="trip_duration")
y_pred = model.predict(X_test)

# --- Evaluate ---
r2_scoring = r2_score(y_test, y_pred)
print(f"R² (Test): {r2_scoring:.4f}")


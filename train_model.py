# train_model.py
# -------------------------------
# Train & Save Ridge Regression Model for Taxi Trip Duration Prediction
# -------------------------------

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# -------------------------
# Feature Engineering
# -------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transformations to raw taxi trip data."""
    df = df.copy()

    # Log-transform target if present
    if "trip_duration" in df.columns:
        df["trip_duration"] = np.log1p(df["trip_duration"])

    # Drop useless IDs
    if "id" in df.columns:
        df = df.drop(columns="id")

    # Vendor binary encoding
    if "vendor_id" in df.columns:
        df["is_vendor2"] = (df["vendor_id"] == 2).astype(int)
        df = df.drop(columns="vendor_id")

    # Drop store_and_fwd_flag
    if "store_and_fwd_flag" in df.columns:
        df = df.drop(columns="store_and_fwd_flag")

    # Passenger outliers
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

    # DateTime features
    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["pickup_datetime_month"] = df["pickup_datetime"].dt.month
        df["pickup_datetime_day"] = df["pickup_datetime"].dt.day
        df["pickup_datetime_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_datetime_day_of_week"] = df["pickup_datetime"].dt.dayofweek
        df["pickup_datetime_is_weekend"] = df["pickup_datetime_day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding
        df["pickup_datetime_hour_sin"] = np.sin(2 * np.pi * df["pickup_datetime_hour"] / 24)
        df["pickup_datetime_hour_cos"] = np.cos(2 * np.pi * df["pickup_datetime_hour"] / 24)
        df["pickup_datetime_dow_sin"] = np.sin(2 * np.pi * df["pickup_datetime_day_of_week"] / 7)
        df["pickup_datetime_dow_cos"] = np.cos(2 * np.pi * df["pickup_datetime_day_of_week"] / 7)

        df = df.drop(columns="pickup_datetime")

    # Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * np.arcsin(np.sqrt(a)) * 6371  # Earth radius in km

    if all(col in df.columns for col in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]):
        df["trip_distance_km"] = haversine_distance(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"]
        )
        df["trip_distance_km"] = np.log1p(df["trip_distance_km"])

    # Rush hour features
    if "pickup_datetime_hour" in df.columns and "pickup_datetime_day_of_week" in df.columns:
        df["is_rush_hour"] = df["pickup_datetime_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df["is_weekday_rush"] = (
            (df["pickup_datetime_day_of_week"].isin([0, 1, 2, 3, 4])) &
            (df["pickup_datetime_hour"].isin([7, 8, 9, 16, 17, 18]))
        ).astype(int)
        df["rush_level"] = 0
        df.loc[df["pickup_datetime_hour"].isin([6, 10, 15, 19]), "rush_level"] = 1
        df.loc[(df["pickup_datetime_day_of_week"].isin([0, 1, 2, 3, 4])) &
               (df["pickup_datetime_hour"].isin([7, 8, 9, 16, 17, 18])), "rush_level"] = 2

    # Distance interactions
    if "trip_distance_km" in df.columns:
        if "is_vendor2" in df.columns:
            df["dist_vendor2"] = df["trip_distance_km"] * df["is_vendor2"]
        if "passenger_count" in df.columns:
            df["dist_passenger"] = df["trip_distance_km"] * df["passenger_count"]

        df["dist_pow_2"] = df["trip_distance_km"] ** 2
        df["dist_pow_3"] = df["trip_distance_km"] ** 3
        df["log_dist"] = np.log1p(df["trip_distance_km"])

    return df


# -------------------------
# Load Train & Validation Data
# -------------------------
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("val.csv")

# Apply Feature Engineering
feature_engineering_transformer = FunctionTransformer(feature_engineering, validate=False)
train_transformed = feature_engineering_transformer.fit_transform(train_data)
val_transformed = feature_engineering_transformer.transform(val_data)

# Separate Target
y_train = train_transformed.pop("trip_duration")
y_val = val_transformed.pop("trip_duration")
X_train = train_transformed
X_val = val_transformed


# -------------------------
# Grid Search for Best Alpha
# -------------------------
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("ridge", Ridge(random_state=14))
])

param_grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best alpha:", grid.best_params_)
print("Best CV R²:", grid.best_score_)


# -------------------------
# Retrain on (train + val)
# -------------------------
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

final_model = Ridge(alpha=grid.best_params_["ridge__alpha"], random_state=14)
final_pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("ridge", final_model)
])

final_pipeline.fit(X_trainval, y_trainval)


# -------------------------
# Evaluate with R² only
# -------------------------
print(f"R² (Train): {r2_score(y_train, final_pipeline.predict(X_train)):.4f}")
print(f"R² (Val):   {r2_score(y_val, final_pipeline.predict(X_val)):.4f}")
print(f"R² (Train+Val): {r2_score(y_trainval, final_pipeline.predict(X_trainval)):.4f}")


# -------------------------
# Save Full Pipeline
# -------------------------
joblib.dump(final_pipeline, "ridge_pipeline.pkl")
print("Full pipeline (scaler + model) saved as ridge_pipeline.pkl")

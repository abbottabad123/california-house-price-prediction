"""
House Price Prediction
Dataset: California Housing (1990 census, 20,640 block groups) - real-world data
Models: Linear Regression (interpretable) + Random Forest (higher accuracy)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# 1. Load data
df = pd.read_csv("/home/claude/housing.csv")
df = df.dropna()  # a few rows have missing total_bedrooms

# one-hot encode the only categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

print("Dataset shape:", df.shape)
print(df.head())

y = df["median_house_value"]
X = df.drop(columns=["median_house_value"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Linear Regression (for interpretable coefficients used in the JS calculator)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

lin_mae = mean_absolute_error(y_test, lin_pred)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
lin_r2 = r2_score(y_test, lin_pred)

print("\n--- Linear Regression ---")
print(f"MAE:  ${lin_mae:,.0f}")
print(f"RMSE: ${lin_rmse:,.0f}")
print(f"R2:   {lin_r2:.3f}")

coefs = dict(zip(X.columns, lin_model.coef_))
print("Intercept:", lin_model.intercept_)
print("Coefficients:", coefs)

# 3. Random Forest (higher accuracy, non-linear)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Random Forest ---")
print(f"MAE:  ${rf_mae:,.0f}")
print(f"RMSE: ${rf_rmse:,.0f}")
print(f"R2:   {rf_r2:.3f}")

importances = dict(zip(X.columns, rf_model.feature_importances_))
print("Feature importances:", sorted(importances.items(), key=lambda x: -x[1]))

# 4. Export everything needed for the JS live calculator + report
export = {
    "n_rows": int(len(df)),
    "feature_order": list(X.columns),
    "feature_ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].mean())] for c in X.columns if X[c].dtype != bool},
    "linear_model": {
        "intercept": float(lin_model.intercept_),
        "coefficients": {k: float(v) for k, v in coefs.items()},
        "mae": float(lin_mae), "rmse": float(lin_rmse), "r2": float(lin_r2)
    },
    "random_forest": {
        "mae": float(rf_mae), "rmse": float(rf_rmse), "r2": float(rf_r2),
        "feature_importances": {k: float(v) for k, v in importances.items()}
    }
}

with open("/home/claude/model_export.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nSaved model_export.json")

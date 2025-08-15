# === Inverse Surrogate Model: Predict Parameters from Pressure Curve and Burn Time ===

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
from pathlib import Path

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from master_thesis.my_functions.helper_functions import load_variable_length_csv

# === Paths ===
doe_dir = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs"
params_path = os.path.join(doe_dir, "DOE_data/params_train.csv")
pressures_path = Path(doe_dir) / "DOE_data/pressure_curves_train.csv"
times_path = os.path.join(doe_dir, "DOE_data/time_curves_train.csv")
model_param_path = os.path.join(doe_dir, "RF_model_params.pkl")
scaler_input_save_path = os.path.join(doe_dir, "RF_input_scaler.pkl")
scaler_output_save_path = os.path.join(doe_dir, "RF_output_scaler.pkl")

# === Parameters ===
FIXED_LENGTH = 100

# === Load Data ===
X = pd.read_csv(params_path)  # Now parameters are the inputs
pressure_curves = pd.DataFrame(load_variable_length_csv(pressures_path))
time_curves = pd.DataFrame(load_variable_length_csv(times_path))

# === Filter valid rows (no NaNs in first three entries of pressure curve) ===
valid_rows = ~(pressure_curves.iloc[:, :3].isna().any(axis=1))
X = X[valid_rows].reset_index(drop=True)
pressure_curves = pressure_curves[valid_rows].reset_index(drop=True)
time_curves = time_curves[valid_rows].reset_index(drop=True)

# === Interpolate curves and calculate burn time ===
Y_features = []
for t_vec, p_vec in zip(time_curves.values, pressure_curves.values):
    valid = ~(np.isnan(t_vec) | np.isnan(p_vec))
    t_valid = t_vec[valid]
    p_valid = p_vec[valid]
    if len(t_valid) < 2:
        continue
    interp_func = interp1d(t_valid, p_valid, kind='linear', fill_value="extrapolate")
    new_t = np.linspace(t_valid[0], t_valid[-1], FIXED_LENGTH)
    new_p = interp_func(new_t)
    burn_time = t_valid[-1] - t_valid[0]
    Y_features.append(np.concatenate([new_p, [burn_time]]))

Y_features = np.array(Y_features)
X = X.iloc[:len(Y_features)]  # Adjust input size to match Y

# === Normalize Inputs and Outputs ===
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y_features)

# === Train/Test Split ===
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, Y_train)

# === Save Model and Scalers ===
joblib.dump(model, model_param_path)
joblib.dump(scaler_X, scaler_input_save_path)
joblib.dump(scaler_Y, scaler_output_save_path)
print("✅ Inverse model and scalers saved.")

# === Evaluate ===
score = model.score(X_test, Y_test)
print(f"R^2 score on pressure curve prediction: {score:.3f}")

# === Prediction Example ===
predicted_curve = scaler_Y.inverse_transform(model.predict([X_scaled[0]]))[0]
true_curve = Y_scaled[0]
print("\nPredicted Curve (first 5 values):\n", predicted_curve[:5])
print("\nTrue Curve (first 5 values):\n", scaler_Y.inverse_transform([true_curve])[0][:5])

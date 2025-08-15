

# === Inverse Surrogate Model (GPR): Predict Parameters from Pressure Curve and Burn Time ===

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from master_thesis.my_functions.helper_functions import load_variable_length_csv

# === Paths ===
doe_dir = "master_thesis/my_tests/10_test_classification_doe/endburner/DOE_outputs"
params_path = os.path.join(doe_dir, "params.csv")
pressures_path = os.path.join(doe_dir, "pressure_curves.csv")
times_path = os.path.join(doe_dir, "time_curves.csv")
model_param_path = os.path.join(doe_dir, "GPR_model_params.pkl")
scaler_input_save_path = os.path.join(doe_dir, "GPR_input_scaler.pkl")
scaler_output_save_path = os.path.join(doe_dir, "GPR_output_scaler.pkl")

FIXED_LENGTH = 100

# === Load and preprocess ===
Y = pd.read_csv(params_path)
pressure_curves = pd.DataFrame(load_variable_length_csv(pressures_path))
time_curves = pd.DataFrame(load_variable_length_csv(times_path))

valid_rows = ~(pressure_curves.iloc[:, :3].isna().any(axis=1))
Y = Y[valid_rows].reset_index(drop=True)
pressure_curves = pressure_curves[valid_rows].reset_index(drop=True)
time_curves = time_curves[valid_rows].reset_index(drop=True)

X_features = []
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
    X_features.append(np.concatenate([new_p, [burn_time]]))

X_features = np.array(X_features)
Y = Y.iloc[:len(X_features)]

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_features)
Y_scaled = scaler_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

kernel = C(1.0) * RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
model.fit(X_train, Y_train)

joblib.dump(model, model_param_path)
joblib.dump(scaler_X, scaler_input_save_path)
joblib.dump(scaler_Y, scaler_output_save_path)
print("✅ GPR model and scalers saved.")

score = model.score(X_test, Y_test)
print(f"R^2 score on GPR model: {score:.3f}")
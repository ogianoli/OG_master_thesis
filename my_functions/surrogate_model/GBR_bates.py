# === Inverse Surrogate Model (GBR): Predict Parameters from Pressure Curve and Burn Time ===

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from master_thesis.my_functions.helper_functions import load_variable_length_csv
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from pathlib import Path





# === Paths ===
doe_dir = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs"
params_path = os.path.join(doe_dir, "params.csv")
# pressures_path = os.path.join(doe_dir, "pressure_curves.csv")
pressures_path = Path(doe_dir) / "pressure_curves.csv"
times_path = os.path.join(doe_dir, "time_curves.csv")
model_param_path = os.path.join(doe_dir, "GBR_model_params.pkl")
scaler_input_save_path = os.path.join(doe_dir, "GBR_input_scaler.pkl")
scaler_output_save_path = os.path.join(doe_dir, "GBR_output_scaler.pkl")

FIXED_LENGTH = 100

X = pd.read_csv(params_path)
print(pressures_path)
pressure_curves = pd.DataFrame(load_variable_length_csv(pressures_path))
time_curves = pd.DataFrame(load_variable_length_csv(times_path))

valid_rows = ~(pressure_curves.iloc[:, :3].isna().any(axis=1))
X = X[valid_rows].reset_index(drop=True)
pressure_curves = pressure_curves[valid_rows].reset_index(drop=True)
time_curves = time_curves[valid_rows].reset_index(drop=True)

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

Y = np.array(Y_features)
X = X.iloc[:len(Y)]

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

X_train_full, X_test, Y_train_full, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
X_train = X_train_full[:len(X_train_full) // 10]
Y_train = Y_train_full[:len(Y_train_full) // 10]

base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
model = MultiOutputRegressor(base_model)

# Simulated progressive training on half of the data (not efficient, just for progress bar)
# for i in tqdm(range(len(X_train)), desc="Training GBR model"):
model.fit(X_train, Y_train)

joblib.dump(model, model_param_path)
joblib.dump(scaler_X, scaler_input_save_path)
joblib.dump(scaler_Y, scaler_output_save_path)
print("✅ GBR model and scalers saved.")

score = model.score(X_test, Y_test)
print(f"R^2 score on GBR model: {score:.3f}")


# # === Track training loss manually ===
# training_losses = []
# for i in tqdm(range(1, len(X_train) + 1), desc="Training GBR model with loss tracking"):
#     model.fit(X_train[:i], Y_train[:i])
#     pred = model.predict(X_train[:i])
#     loss = mean_squared_error(Y_train[:i], pred)
#     training_losses.append(loss)

# # === Plot Loss Over Iterations ===
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o')
# plt.title("Training Loss over Iterations")
# plt.xlabel("Number of Samples")
# plt.ylabel("Mean Squared Error")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # === Plot Predicted vs. True Parameters ===
# Y_test_pred = model.predict(X_test)
# Y_test_true = scaler_Y.inverse_transform(Y_test)
# Y_test_pred_inv = scaler_Y.inverse_transform(Y_test_pred)

# num_params = Y.shape[1]
# param_names = Y.columns if hasattr(Y, 'columns') else [f"Param {i+1}" for i in range(num_params)]

# fig, axs = plt.subplots(nrows=num_params, figsize=(8, 4 * num_params))
# if num_params == 1:
#     axs = [axs]

# for i in range(num_params):
#     axs[i].scatter(Y_test_true[:, i], Y_test_pred_inv[:, i], alpha=0.5)
#     axs[i].plot([Y_test_true[:, i].min(), Y_test_true[:, i].max()],
#                 [Y_test_true[:, i].min(), Y_test_true[:, i].max()],
#                 color='red', linestyle='--')
#     axs[i].set_title(f"Test Set: {param_names[i]}")
#     axs[i].set_xlabel("True")
#     axs[i].set_ylabel("Predicted")
#     axs[i].grid(True)

# plt.tight_layout()
# plt.show()
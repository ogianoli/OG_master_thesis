import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from master_thesis.my_functions.helper_functions import *

# -------------------------
# Load test data
params_test_path = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/params_test.csv"
pressure_curves_test_path = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/pressure_curves_test.csv"
model_param_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/RF_model_params.pkl"
scaler_input_save_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/RF_input_scaler.pkl"
scaler_output_save_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/RF_output_scaler.pkl"
plot_dir = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/plots"
pivot_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/plots/pivot_table.csv"

INTERP_POINTS = 100  # number of time steps to normalize to

def interpolate_curve(curve, target_length=INTERP_POINTS):
    x = np.linspace(0, 1, len(curve))
    f = interp1d(x, curve, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

# === Load Data ===
params_test = pd.read_csv(params_test_path)
pressure_curves_test_raw = pd.DataFrame(load_variable_length_csv(pressure_curves_test_path))

# === Load Model and Scalers ===
model = joblib.load(model_param_path)
scaler_input = joblib.load(scaler_input_save_path)
scaler_output = joblib.load(scaler_output_save_path)

# === Predict Scaled and Inverse Transformed Curves ===
scaled_inputs = scaler_input.transform(params_test.values)
predicted_scaled = model.predict(scaled_inputs)
predicted_curves_raw = scaler_output.inverse_transform(predicted_scaled)

# Interpolate all target and predicted curves, tracking valid rows
predicted_curves_interp = np.array([interpolate_curve(c) for c in predicted_curves_raw])
target_curves_interp = []
valid_indices = []
for i, (_, row) in enumerate(pressure_curves_test_raw.iterrows()):
    raw_vals = row.dropna().values
    if len(raw_vals) > 1 and not np.isnan(raw_vals[:3]).all():
        target_curves_interp.append(interpolate_curve(raw_vals))
        valid_indices.append(i)
target_curves_interp = np.array(target_curves_interp)
params_test = params_test.iloc[valid_indices].reset_index(drop=True)
predicted_curves_interp = predicted_curves_interp[valid_indices]

print(f"✅ Valid test samples after filtering: {len(valid_indices)}")

# === Compute R² errors ===
r2_errors = np.array([
    r2_score(target_curves_interp[i], predicted_curves_interp[i])
    for i in range(len(target_curves_interp))
])
params_test["r2_error"] = r2_errors


# Bin 'diameter' and 'length' into 4 ranges each
params_test["diameter_bin"] = pd.qcut(params_test["diameter"], 4, labels=False, duplicates='drop')
params_test["length_bin"] = pd.qcut(params_test["length"], 4, labels=False, duplicates='drop')

os.makedirs(plot_dir, exist_ok=True)

# Generate 16 plots for each (diameter_bin, length_bin) combination
for d_bin in range(4):
    for l_bin in range(4):
        subset = params_test[(params_test["diameter_bin"] == d_bin) & (params_test["length_bin"] == l_bin)]

        mean_r2 = subset['r2_error'].mean()
        min_r2 = subset['r2_error'].min()
        max_r2 = subset['r2_error'].max()
        print(f"🧪 Bin ({d_bin},{l_bin}): {len(subset)} samples | Mean R²: {mean_r2:.4f} | Min: {min_r2:.4f} | Max: {max_r2:.4f}")
        print(f" - Unique throats: {subset['throat'].nunique()}, Unique coreDiameters: {subset['coreDiameter'].nunique()}")

        if len(subset) < 3:
            continue  # Not enough data to plot

        # Bin 'throat' and 'coreDiameter' into 20 bins each for density
        subset["throat_bin"] = pd.cut(subset["throat"], bins=20)
        subset["coreDiameter_bin"] = pd.cut(subset["coreDiameter"], bins=20)

        df = pd.DataFrame({
            "throat_bin": subset["throat_bin"],
            "coreDiameter_bin": subset["coreDiameter_bin"],
            "r2": subset["r2_error"]
        })

        # Create pivot table and interpolate missing values to reduce NaNs
        pivot_table = df.pivot_table(index="coreDiameter_bin", columns="throat_bin", values="r2", aggfunc="mean", fill_value=np.nan)
        pivot_table.index = pivot_table.index.map(lambda x: x.mid)
        pivot_table.columns = pivot_table.columns.map(lambda x: x.mid)
        pivot_table = pivot_table.interpolate(method="linear", axis=0).interpolate(method="linear", axis=1)

        # Save the first valid pivot table to CSV
        if d_bin == 0 and l_bin == 0:
            pivot_table.to_csv(pivot_path)
            print(f"💾 Saved pivot table to: {pivot_path}")

        # Only plot if grid is meaningful
        if pivot_table.shape[0] > 1 and pivot_table.shape[1] > 1:
            print(f"📊 Plotting surface for bin ({d_bin},{l_bin}) with shape {pivot_table.shape}")
            plot_3d_error_surface(
                x_vals=pivot_table.columns.values,
                y_vals=pivot_table.index.values,
                error_matrix=pivot_table.values,
                x_label="Throat",
                y_label="Core Diameter",
                z_label="R² Score",
                title=f"R2 Error (diameter_bin={d_bin}, length_bin={l_bin})",
                save_dir=plot_dir,
                contour=True
            )
        else:
            print(f"⚠️ Skipping plot for bin ({d_bin},{l_bin}) - insufficient grid shape: {pivot_table.shape}")
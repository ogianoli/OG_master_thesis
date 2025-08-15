# Helper functions for the SRM simulation
import joblib
import numpy as np
import os
import sys
import subprocess
import pandas as pd
import yaml
# from ruamel.yaml import YAML
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import h5py
# import torch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from master_thesis.openmotor.uilib import fileIO

# set the path to the parent directory
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from master_thesis.openmotor.motorlib import motor
from master_thesis.openmotor.motorlib.grain import PerforatedGrain as Grain
from master_thesis.openmotor.motorlib.propellant import Propellant
from master_thesis.openmotor.motorlib.constants import gasConstant
from scipy.interpolate import interp1d
# Get parameters function from .ric file
# def getParameters(motor, param_name):


# Synthetic SRM-like pressure curve function
def generate_srm_pressure_curve(time_vector, peak_pressure=200):
    rise_time = 1.0
    plateau_time = 3.5
    fall_time = time_vector[-1] - rise_time - plateau_time

    pressure = np.zeros_like(time_vector)
    
    for i, t in enumerate(time_vector):
        if t < rise_time:
            pressure[i] = (t / rise_time) * peak_pressure
        elif t < rise_time + plateau_time:
            pressure[i] = peak_pressure * (1 - 0.05 * (t - rise_time) / plateau_time)
        elif t <= time_vector[-1]:
            decline_ratio = (t - rise_time - plateau_time) / fall_time
            pressure[i] = peak_pressure * (0.95 * (1 - decline_ratio))
    
    return pressure


def delta_fn(r):
    """
    Function to compute burning thickness based on burn distance.

    Parameters:
        r_i : burn distance [m]
    """
    return Grain.isWebLeft(r)  # Assuming this is a function that returns the burning thickness


def compute_KN_single_grain(D_out, D_throat, reg):
    D_core      = D_out/2
    r          = reg
    N_faces    = 0
    
    """
    Computes K_N for a single grain geometry.

    Parameters:
        D_out      : outer diameter of the grain [m]
        D_core     : core diameter [m]
        r          : burn distance [m]
        z_start_fn : function z_start(r)
        z_end_fn   : function z_end(r)
        N_faces    : number of burning faces (0, 1, or 2)
        delta_fn   : function delta(r), burning thickness [m]
        D_throat   : throat diameter [m]

    Returns:
        KN         : geometry ratio (unitless)
    """
    port_area = np.pi * (D_core + 2 * r) * (Grain.getEndPositions(r)[1] - Grain.getEndPositions(r)[0])
    face_area = N_faces * np.pi * ((D_out / 2) - ((D_core + 2 * r) / 2))
    total_area = (port_area + face_area) * delta_fn(r)

    throat_area = np.pi * (D_throat / 2) ** 2
    KN = total_area / throat_area
    return KN


# def compute_P_chamber(D_out, D_throat):
#     """
#     Computes chamber pressure for a single grain motor.

#     Parameters:
#         KN     : geometry ratio (unitless)
#         rho    : propellant density [kg/m³]
#         a      : burn rate coefficient
#         gamma  : ratio of specific heats
#         R      : universal gas constant [J/(kmol·K)]
#         M      : molar mass of gas [kg/kmol]
#         T      : chamber temperature [K]
#         n      : burn rate exponent

#     Returns:
#         P_chamber : chamber pressure (units depend on input)
#     """

#     my_prop = Propellant()
#     R = gasConstant
#     print(Propellant.getProperty(prop, 'density'))
#     tab  = Propellant.getProperty('tabs')
#     a, n, gamma, T, M = tab['a'], tab['n'], tab['k'], tab['t'], tab['m']
#     print("a, n, gamma, T, M: ", a, n, gamma, T, M)
#     c_star = np.sqrt(gamma / ((R / M) * T)) * ((2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1))))
#     P_chamber = ((compute_KN_single_grain(D_out, D_throat, reg) * rho * a) / c_star) ** (1 / (1 - n))
#     return P_chamber


def run_openmotor(motor_path, saving_path): # Run openmotor with current ric file, works

    # Define the command as a list
    command = ["python", "master_thesis/openmotor/main.py", "-o", saving_path, "-h", motor_path]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Print the output
    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

def get_pressure_time_func(saving_path):
    # Read into a DataFrame
    df = pd.read_csv(saving_path)
    # Check actual column names
    print("Columns:", df.columns.tolist())
    # Extract vectors
    time_vector = df["Time(s)"].values
    pressure_vector = df["Chamber Pressure(Pa)"].values
    # Store into a numpy array
    result_array = np.column_stack((time_vector, pressure_vector))
    # Optional: print to verify
    print(result_array)

    return result_array



def get_thrust_time_func(saving_path):
    # Read into a DataFrame
    df = pd.read_csv(saving_path)
    # Check actual column names
    print("Columns:", df.columns.tolist())
    # Extract vectors
    time_vector = df["Time(s)"].values
    pressure_vector = df["Thrust(N)"].values
    # Store into a numpy array
    result_array = np.column_stack((time_vector, pressure_vector))
    # Optional: print to verify
    print(result_array)

    return result_array

def get_array(saving_path, key):
    # Read into a DataFrame
    df = pd.read_csv(saving_path)
    # Check actual column names
    # print("Columns:", df.columns.tolist())
    # Extract vectors
    vector = df[key].values
    # Store into a numpy array
    result_array = np.array(vector)
    # Optional: print to verify
    # print(result_array)

    return result_array

# def update_motor_file(motor_file, x1, x2):
#     # Load the motor file using OpenMotor's custom loader
#     motor_data = fileIO.loadFile(motor_file, fileIO.fileTypes.MOTOR)

#     # Modify grain and nozzle parameters
#     motor_data["grains"][0]["properties"]["diameter"] = float(x1)
#     motor_data["nozzle"]["throat"] = float(x2)

#     # Save it back using OpenMotor's custom saver
#     fileIO.saveFile(motor_file, motor_data, fileIO.fileTypes.MOTOR)



def update_motor_file(motor_file, **params):
    with open(motor_file, "r") as f:
        content = f.read()

    for key, value in params.items():
        pattern = rf"({key}:\s*)([0-9.eE+-]+)"
        content = re.sub(pattern, lambda m: m.group(1) + str(value), content)

    with open(motor_file, "w") as f:
        f.write(content)


def normalize(sim_pressure, target_pressure):
    """
    Normalize the sim_pressure to match target_pressure length.
    """
    # Check if lengths are different
    if len(sim_pressure) != len(target_pressure):
        # Interpolate sim_pressure to match target_pressure length
        x = np.arange(len(sim_pressure))
        f = interp1d(x, sim_pressure, kind='linear', fill_value="extrapolate")
        sim_pressure = f(np.linspace(0, len(sim_pressure)-1, len(target_pressure)))
    # print("Lengths after normalization: (sim_preeure)(target_pressure)")
    # print(len(sim_pressure), len(target_pressure))
    return sim_pressure

# === DIRECTORY HANDLING FUNCTIONS ===

def get_valid_data_dir(data_num, base_path="data", prefix="opt"):
    while True:
        try:
            # data_num = int(input("Enter a data number: ").strip())
            data_dir = os.path.join(base_path, f"{prefix}{data_num}")
            user_input = input(f"The directory is: '{data_dir}'.Do you want to add something special? (yes/no): ").strip()
            while user_input.lower() not in ["yes", "no"]:
                print("⚠️ Please answer with 'yes' or 'no'.")
                user_input = input(f"The directory is: '{data_dir}'.Do you want to add something special? (yes/no): ").strip()
            
            if user_input.lower() == "yes":
                    data_dir = os.path.join(data_dir, input("Enter the special name: ").strip())
            elif user_input.lower() == "no":
                    pass
            
            if os.path.exists(data_dir):
                user_input = input(f"The directory '{data_dir}' already exists. Overwrite it? (yes/no): ").strip().lower()
                if user_input == "yes":
                    shutil.rmtree(data_dir)  # Delete everything inside
                    os.makedirs(data_dir)
                    print(f"✅ Overwritten and recreated: '{data_dir}'")
                    return data_dir
                elif user_input == "no":
                    print("Please choose another number.")
                    continue
                else:
                    print("⚠️ Please answer with 'yes' or 'no'.")
                    continue
            else:
                os.makedirs(data_dir)
                print(f"✅ Created new directory: '{data_dir}'")
                return data_dir

        except ValueError:
            print("❌ Invalid input. Please enter a valid integer for data number.")


def get_next_data_num(base_path="data", prefix="opt"):
    max_num = 0
    pattern = re.compile(rf"^{prefix}(\d+)$")
    user_input = input(f"Want to overwrite something (yes), or chose next number (no)? ").strip().lower()
    while user_input.lower() not in ["yes", "no"]:
        print("⚠️ Please answer with 'yes' or 'no'.")
        user_input = input(f"Want to overwrite something (yes), or chose next number (no)? ").strip().lower()
    if user_input == "yes":
        data_num = input(f"digit number to overwrite: ").strip().lower()
        while not data_num.isdigit():
            print("⚠️ Please enter a valid integer.")
            data_num = input(f"digit number to overwrite: ").strip().lower()
        return data_num
    elif user_input == "no":
        print("Selecting the next number...")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for name in os.listdir(str(base_path)):
            match = pattern.match(name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

        return max_num + 1


# === FILE SAVING FUNCTIONS ===
    
def save_run_info(data_dir, data_num, time_vector, motor_file, best_x=None, best_f=None):
    """Saves info like time_step, motor_file path and comment to a .txt file"""
    time_step = round(np.mean(np.diff(time_vector)), 6)
    comment = input("💬 Enter a short comment for this run (or leave blank): ").strip()

    info_path = os.path.join(data_dir, f"info{data_num}.txt")
    with open(info_path, "w") as f:
        f.write(f"time_step: {time_step}\n")
        f.write(f"motor_file: {motor_file}\n")
        f.write(f"comment: {comment if comment else '[none]'}\n")
        if best_x is not None and best_f is not None:
            best_x_str = ", ".join(f"{val:.6f}" for val in best_x)
            f.write(f"best_x: {best_x_str}\n")
            f.write(f"best_error: {best_f[0]:.6e}\n")
    
    print(f"✅ Saved run info to: '{info_path}'")

def save_pressure_curves(curves, filepath):
    """Save pressure curves to a CSV file.

    Args:
        curves (list of list of floats): Each curve is a list of pressure values.
        filepath (str): Path to the output CSV file.
    """
    with open(filepath, "w") as f:
        for curve in curves:
            f.write(",".join(f"{p:.6f}" for p in curve) + "\n")
    print(f"✅ Saved {len(curves)} pressure curves to: '{filepath}'")

def save_time_vectors(time_curves, filepath):
    """Save time vectors (one per row) to a CSV file.

    Args:
        time_curves (list of list of float): List of time arrays per iteration.
        filepath (str): Path to the output CSV file.
    """
    with open(filepath, "w") as f:
        for curve in time_curves:
            f.write(",".join(f"{t:.6f}" for t in curve) + "\n")
    print(f"✅ Saved {len(time_curves)} time curves to: '{filepath}'")


def save_metadata(params, errors, success_flags, filepath, optimization_variables=None):
    """Save optimization metadata to a CSV file.

    Args:
        params (list of tuples): List of (diameter, throat) pairs.
        errors (list of floats): Mean squared error for each iteration.
        success_flags (list of str): 'yes' or 'no' flags for simulation success.
        filepath (str): Output path for the metadata CSV.
    """
    # Support dynamic number of parameters
    data_dict = {"Iteration": list(range(1, len(params) + 1)), "Error": errors, "Sim_Success": success_flags}
    
    for i, var in enumerate(optimization_variables):
        var_name = var["name"]
        data_dict[var_name] = [p[i] for p in params]
    
    df_meta = pd.DataFrame(data_dict)
    df_meta.to_csv(filepath, index=False)
    print(f"✅ Saved metadata to: '{filepath}'")


    # === PLOTTING FUNCTIONS ===

def get_color(index, start_idx, total, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=start_idx, vmax=start_idx + total - 1)
    return cmap(norm(index))

def plot_pressure_curves(ax, pressures, times, time_vector, target_pressure, final_time, final_pressure,
                         best_x, best_f, pop_size, optimization_variables, yellow_end=10, red_end=50):
    total_iters = len(pressures)
    
    def count_in_range(start_gen, end_gen):
        return sum(start_gen <= (i // pop_size) <= end_gen for i in range(total_iters))
    
    yellow_count = count_in_range(0, yellow_end)
    red_count = count_in_range(yellow_end + 1, red_end)
    blue_count = total_iters - yellow_count - red_count

    for i, (pressure, time) in enumerate(zip(pressures, times)):
        gen_num = i // pop_size
        if gen_num <= yellow_end:
            color = get_color(i, 0, yellow_count, 'YlOrBr')
        elif gen_num <= red_end:
            color = get_color(i, yellow_count, red_count, 'Reds')
        else:
            color = get_color(i, yellow_count + red_count, blue_count, 'Blues')
        ax.plot(time, pressure, color=color, alpha=0.8, linewidth=1)

    # Best fit and target
    ax.plot(final_time, final_pressure, label='Last params', color='black', linewidth=2)
    ax.plot(time_vector, target_pressure, label='Target Pressure', linestyle='--', color='orange', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Pressure (Pa)', fontsize=14)
    ax.set_title('Pressure Curve Fit Over Iterations', fontsize=16)
    ax.grid(True)

    # Legend
    legend_text = "Color legend:\nY: Gen 0–10\nR: Gen 11–50\nB: Gen 51+"
    ax.text(0.97, 0.97, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.6))

    # Best solution info
    textstr = "Best variables:\n" + "\n".join(
        f"{v['label'].split()[0]}: {val:.5f}" for v, val in zip(optimization_variables, best_x)
    ) + f"\nError = {best_f[0]:.2e}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, bbox=props, verticalalignment='top')

def plot_error(ax, errors):
    ax.plot(range(1, len(errors) + 1), errors, marker='o', color='crimson', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('R2 Error', fontsize=14)
    ax.set_title('Error Over Iterations', fontsize=16)
    ax.grid(True)

def plot_param_evolution(ax, values, title, ylabel, color='teal'):
    ax.plot(range(1, len(values) + 1), values, color=color, marker='o', markersize=3, linewidth=1)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True)

def plot_all(
    iteration_curves_pressure,
    iteration_curves_time,
    iteration_errors,
    iteration_params,
    time_vector,
    target_pressure,
    final_sim_time,
    final_sim_pressure,
    best_x,
    best_f,
    pop_size,
    plot_dir,
    optimization_variables,
    gen_range=None
):
    # Optionally filter errors, params, and curves based on gen_range
    if gen_range:
        start_gen, end_gen = gen_range
        start_idx = start_gen * pop_size
        end_idx = (end_gen + 1) * pop_size

        iteration_errors = iteration_errors[start_idx:end_idx]
        iteration_params = iteration_params[start_idx:end_idx]
        iteration_curves_pressure = iteration_curves_pressure[start_idx:end_idx]
        iteration_curves_time = iteration_curves_time[start_idx:end_idx]

    # Improved layout using matplotlib.gridspec
    import matplotlib.gridspec as gridspec
    num_vars = len(optimization_variables)
    fig = plt.figure(figsize=(18, 10), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(2, max(4, num_vars), height_ratios=[3, 2], hspace=0.35, wspace=0.3)

    # Top row: pressure and error plots (make them span more columns to appear wider)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Bottom row: parameter evolution plots
    param_axes = []
    for i in range(num_vars):
        ax = fig.add_subplot(gs[1, i])
        param_axes.append(ax)

    # Pressure curves plot
    plot_pressure_curves(
        ax1, iteration_curves_pressure, iteration_curves_time,
        time_vector, target_pressure, final_sim_time,
        final_sim_pressure, best_x, best_f, pop_size, optimization_variables
    )

    # Error plot
    plot_error(ax2, iteration_errors)

    # Parameter evolution (dynamic subplots)
    param_names = [var['label'] for var in optimization_variables]
    param_colors = [var.get('color', 'teal') for var in optimization_variables]
    for i, (name, color) in enumerate(zip(param_names, param_colors)):
        values = [p[i] for p in iteration_params]
        plot_param_evolution(param_axes[i], values, f"{name} Evolution", name, color=color)

    # Hide unused axes if num_vars < total_cols - 1 (because ax1, ax2 occupy [0,1] and [0,2])
    total_cols = max(3, num_vars)
    if num_vars < total_cols - 1:
        for j in range(num_vars, total_cols - 1):
            fig.add_subplot(gs[1, j]).set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_dir, dpi=300)
    print(f"✅ Plot saved to: {plot_dir}")
    plt.show()


def save_runs_to_hdf5(optimization_variables=None):
    print("\n📦 Starting data export to HDF5")

    # Ask user whether to create a new file or use an existing one
    create_new = input("Do you want to create a new HDF5 file? (yes/no): ").strip().lower()

    if create_new == "yes":
        hdf5_path = input("Enter path for new HDF5 file (e.g., data/all_runs.h5): ").strip()
        if os.path.exists(hdf5_path):
            overwrite = input("File already exists. Overwrite? (yes/no): ").strip().lower()
            if overwrite != "yes":
                print("❌ Aborted.")
                return
        hdf = h5py.File(hdf5_path, "w")
    else:
        hdf5_path = input("Enter path to existing HDF5 file: ").strip()
        if not os.path.exists(hdf5_path):
            print("❌ File does not exist.")
            return
        hdf = h5py.File(hdf5_path, "a")  # append mode

    # Ask which runs to save
    start = int(input("Enter start opt# (e.g., 25): "))
    end = int(input("Enter end opt# (e.g., 30): "))

    for num in range(start, end + 1):
        group_name = f"opt{num}"
        if group_name in hdf:
            answer = input(f"Group '{group_name}' exists. Overwrite? (yes/no): ").strip().lower()
            if answer != "yes":
                print(f"⚠️  Skipped {group_name}")
                continue
            del hdf[group_name]

        print(f"\n➕ Adding {group_name}...")
        grp = hdf.create_group(group_name)

        # === Load metadata ===
        metadata_path = f"data/opt{num}/optimization_metadata{num}.csv"
        metadata = pd.read_csv(metadata_path)
        metadata_np = metadata.to_numpy()

        # Identify parameter columns based on optimization_variables
        param_names = [var["name"] for var in (optimization_variables or [])]
        # Find the columns in metadata that match the optimization variable names
        param_cols = [col for col in metadata.columns if col in param_names]
        # Find all other columns for metadata structure
        other_cols = [col for col in metadata.columns if col not in param_cols]
        # Ensure order: Iteration, param_cols..., Error, Sim_Success
        # Try to find "Iteration", "Error", "Sim_Success" columns
        fixed_cols = []
        if "Iteration" in metadata.columns:
            fixed_cols.append("Iteration")
        fixed_cols += param_cols
        if "Error" in metadata.columns:
            fixed_cols.append("Error")
        if "Sim_Success" in metadata.columns:
            fixed_cols.append("Sim_Success")

        # Build dtype dynamically using real parameter names
        dtype_fields = []
        if "Iteration" in metadata.columns:
            dtype_fields.append(("Iteration", "i4"))
        for name in param_cols:
            dtype_fields.append((name, "f8"))
        if "Error" in metadata.columns:
            dtype_fields.append(("Error", "f8"))
        if "Sim_Success" in metadata.columns:
            dtype_fields.append(("Sim_Success", "S10"))
        full_dtype = np.dtype(dtype_fields)

        # Build structured array using real column names
        metadata_struct = []
        for _, row in metadata.iterrows():
            entry = []
            if "Iteration" in metadata.columns:
                entry.append(int(row["Iteration"]))
            for name in param_cols:
                entry.append(float(row[name]))
            if "Error" in metadata.columns:
                entry.append(float(row["Error"]))
            if "Sim_Success" in metadata.columns:
                entry.append(str(row["Sim_Success"]).encode('utf-8'))
            metadata_struct.append(tuple(entry))

        metadata_struct = np.array(metadata_struct, dtype=full_dtype)

        grp.create_dataset("Params", data=metadata_struct)
        # Save attribute names using real column names
        for i, col in enumerate(fixed_cols):
            grp["Params"].attrs[f"col{i}"] = col

        # === Load pressure curves ===
        curve_path = f"data/opt{num}/optimization_curves{num}.csv"
        with open(curve_path, "r") as f:
            lines = f.readlines()
        curves = [
            np.array([float(val) for val in line.strip().split(",") if val.strip() != ""])
            for line in lines if line.strip() != ""
        ]
        
        pressure_grp = grp.create_group("Pressure_Curves")
        for i, curve in enumerate(curves):
            pressure_grp.create_dataset(f"curve_{i:04d}", data=curve)

        # === Load info file ===
        info_path = f"data/opt{num}/info{num}.txt"
        info_dict = {}
        with open(info_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                info_dict[key] = value

        grp.attrs["time_step"] = float(info_dict["time_step"])
        grp.attrs["motor_file"] = info_dict["motor_file"]
        grp.attrs["comment"] = info_dict["comment"]

        print(f"✅ Saved {group_name} to HDF5")

    hdf.close()
    print(f"\n💾 Export complete. File saved at: {hdf5_path}")

def check_variables(optimization_variables, motor_file):
    """Check if all optimization variable names exist in the motor file."""
    with open(motor_file, "r") as f:
        content = f.read()

    missing_vars = []
    for var in optimization_variables:
        var_name = var["name"]
        if var_name not in content:
            missing_vars.append(var_name)

    if missing_vars:
        print("❌ ERROR: The following variable(s) are missing in the motor file:")
        for var in missing_vars:
            print(f" - {var}")
        print("\nPlease correct the variable names and retry.")
        sys.exit(1)
    else:
        print("✅ All variables found in the motor file. Proceeding...")


        import numpy as np
import pandas as pd

def load_variable_length_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    max_len = max(len(line.strip().split(',')) for line in lines)
    pressure_curves = []
    for line in lines:
        values = [float(x) if x else np.nan for x in line.strip().split(',')]
        padded = values + [np.nan] * (max_len - len(values))
        pressure_curves.append(padded)

    return np.array(pressure_curves)

def predict_pressure_curve_from_params(params, model_path, scaler_X_path, scaler_Y_path):
    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)

    params = np.array(params).reshape(1, -1)
    scaled_params = scaler_X.transform(params)
    prediction_scaled = model.predict(scaled_params)
    prediction = scaler_Y.inverse_transform(prediction_scaled)
    return prediction.flatten()

def plot_3d_error_surface(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    error_matrix: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    save_dir: str = "./plots",
    contour: bool = True,
    levels: int = 50,
    cmap: str = 'viridis'
):
    """
    Create and save a contour or surface plot of error vs. two variables.
    
    Parameters:
        x_vals (np.ndarray): 1D array for x-axis values
        y_vals (np.ndarray): 1D array for y-axis values
        error_matrix (np.ndarray): 2D array of error values (shape must match meshgrid of x and y)
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        z_label (str): Label for the colorbar / z-axis
        title (str): Title of the plot (used in filename as well)
        save_dir (str): Directory where the plot will be saved
        contour (bool): If True, creates a contour plot; else, a 3D surface plot
        levels (int): Number of contour levels (for contour plot)
        cmap (str): Colormap for the plot
    """
    # Create meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)

    # Create save path
    filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 7))

    if contour:
        # Use percentiles to ignore outliers and boost contrast
        finite_vals = error_matrix[np.isfinite(error_matrix)]
        if finite_vals.size == 0:
            print("⚠️ No finite values to plot in error matrix.")
            return

        print("📊 Sample error matrix values:")
        print(error_matrix[:5, :5])

        vmin = np.percentile(finite_vals, 1)
        vmax = np.percentile(finite_vals, 99)

        # Enforce reasonable bounds if vmin/vmax are too close
        if abs(vmax - vmin) < 1e-6:
            print("⚠️ vmin and vmax are nearly equal. Forcing scale to [-1, 1]")
            vmin, vmax = -1.0, 1.0

        print(f"🎨 Plot scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

        error_matrix = np.clip(error_matrix, vmin, vmax)

        contour_plot = plt.contourf(X, Y, error_matrix, levels=np.linspace(vmin, vmax, levels), cmap=cmap)
        plt.colorbar(contour_plot, label=z_label)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, error_matrix, cmap=cmap)
        fig.colorbar(surf, ax=ax, label=z_label)
        ax.set_zlabel(z_label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.savefig(save_path)
        plt.close()
        return

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def rotate(points, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return points @ R.T

def mirror_across_line(points, direction):
    direction = direction / np.linalg.norm(direction)
    projection = np.outer(np.dot(points, direction), direction)
    rejection = points - projection
    mirrored = projection - rejection
    return mirrored

def plot_star_geometry(N, Ri, Ro, w, f, e, savefolder = '.',
                       savename = None, show_plot = True):
    """ Function creates a sketch of the star shaped geometry based on the
        parametrization found in e.g. 
        Oh, Seok-Hwan, Tae-Seong Roh, and Hyoung Jin Lee.
        "A Study on the Optimal Design Method for Star-Shaped Solid
        Propellants through a Combination of Genetic Algorithm and
        Machine Learning." Aerospace 10.12 (2023): 979.

        Args:
            N (int): number of star branches
            Ri (float): internal radius
            Ro (float): extrenal radius
            w (float): web thickness
            f (float): fillet radius
            e (float): angle coefficient

        Returns:
            (None): plots / saves the resulting image
    """

    # if savename is not None:
    os.makedirs(savefolder, exist_ok=True)
    if not isinstance(N, int): 
        N = int(N) # just to make sure input is integer
        print(f"Converting input number of star branches to integer {N}")
    Rp = Ro - w - f
    angle1 = np.pi * e / N
    angle2 = np.pi / N
    theta = 2 * np.arctan((Rp * np.sin(angle1) * np.tan(angle1)) /
                          (Rp * np.sin(angle1) - Ri * np.tan(angle1)))
    half_theta = theta / 2

    Line_theta_dir = np.array([np.cos(half_theta), np.sin(half_theta)])
    Line2_dir = np.array([np.cos(angle1), np.sin(angle1)])
    Line4_dir = np.array([np.cos(angle2), np.sin(angle2)])

    Point0 = np.array([0, 0])
    P_Ri = np.array([Ri, 0])
    A = np.array([Line_theta_dir, -Line2_dir]).T
    b = Point0 - P_Ri
    t_values = np.linalg.solve(A, b)
    Point1 = P_Ri + t_values[0] * Line_theta_dir
    Point2 = Point1 + f * Line2_dir

    # Arc1
    angle_start_arc1 = np.arctan2(Point2[1], Point2[0])
    arc1_angles = np.linspace(angle_start_arc1, angle2, 100)
    arc1_pts = np.column_stack([
        (Rp + f) * np.cos(arc1_angles),
        (Rp + f) * np.sin(arc1_angles)
    ])

    # Arc2
    angle_start_arc2 = np.arctan2(Point2[1] - Point1[1], Point2[0] - Point1[0])
    arc2_angles = np.linspace(angle_start_arc2, angle_start_arc2 - 2 * np.pi, 10000)
    arc2_coords = []
    arc2_end = None
    for i in range(1, len(arc2_angles)):
        a1 = arc2_angles[i - 1]
        a2 = arc2_angles[i]
        p1 = Point1 + f * np.array([np.cos(a1), np.sin(a1)])
        p2 = Point1 + f * np.array([np.cos(a2), np.sin(a2)])
        tangent_vec = p2 - p1
        tangent_unit = tangent_vec / np.linalg.norm(tangent_vec)
        angle_diff = np.arccos(np.clip(np.dot(tangent_unit, Line_theta_dir), -1.0, 1.0))
        arc2_coords.append(p2)
        if np.isclose(angle_diff, np.deg2rad(180), atol=1e-2):
            arc2_end = p2
            break
    arc2_coords = np.array(arc2_coords)
    if arc2_end is None:
        arc2_end = arc2_coords[-1]

    # Line5
    t_line5 = -arc2_end[1] / Line_theta_dir[1]
    Point3 = arc2_end + t_line5 * Line_theta_dir
    line5 = np.array([arc2_end, Point3])

    # Arc3
    arc3_angles = np.linspace(0, angle2, 100)
    arc3_pts = np.column_stack([
        Ro * np.cos(arc3_angles),
        Ro * np.sin(arc3_angles)
    ])

    # Geometry to plot (Line1 and Line4 omitted)
    base_segments = {
        "Line5": line5,
        "Arc1": arc1_pts,
        "Arc2": arc2_coords,
        "Arc3": arc3_pts
    }

    mirrored_segments = {
        label: mirror_across_line(seg, Line4_dir)
        for label, seg in base_segments.items()
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_title(f"Geometry with Mirror (N={N})")
    ax.grid(True, linestyle='--', alpha=0.4)

    for i in range(N):
        angle = i * 2 * np.pi / N
        for label, seg in base_segments.items():
            seg_rot = rotate(seg, angle)
            ax.plot(seg_rot[:, 0], seg_rot[:, 1], lw=1, label=label if i == 0 else "")
        for seg in mirrored_segments.values():
            seg_rot = rotate(seg, angle)
            ax.plot(seg_rot[:, 0], seg_rot[:, 1], lw=1)

    # ax.legend()
    if savename is not None:
        plt.savefig(os.path.join(savefolder, savename))
    if show_plot:
        plt.show()
    plt.close()
    # fig.savefig(f"WORK/geometry_circular.png", dpi=300, bbox_inches='tight')

def plot_star_petal(N, Ri, Ro, w, f, e):
    Rp = Ro - w - f  # Rp is now derived from Ro and w
    angle1 = np.pi * e / N
    angle2 = np.pi / N
    theta = 2 * np.arctan((Rp * np.sin(angle1) * np.tan(angle1)) /
                          (Rp * np.sin(angle1) - Ri * np.tan(angle1)))
    half_theta = theta / 2

    # Direction vectors
    Line_theta_dir = np.array([np.cos(half_theta), np.sin(half_theta)])
    Line2_dir = np.array([np.cos(angle1), np.sin(angle1)])
    Line4_dir = np.array([np.cos(angle2), np.sin(angle2)])

    # Base points
    Point0 = np.array([0, 0])
    P_Ri = np.array([Ri, 0])
    A = np.array([Line_theta_dir, -Line2_dir]).T
    b = Point0 - P_Ri
    t_values = np.linalg.solve(A, b)
    Point1 = P_Ri + t_values[0] * Line_theta_dir
    Point2 = Point1 + f * Line2_dir

    # Arc1 from center (Point0)
    angle_start_arc1 = np.arctan2(Point2[1], Point2[0])
    arc1_angles = np.linspace(angle_start_arc1, angle2, 100)
    arc1_radius = Rp + f
    arc1_x = arc1_radius * np.cos(arc1_angles)
    arc1_y = arc1_radius * np.sin(arc1_angles)

    # Arc2 from Point2 (centered at Point1), clockwise, stop when tangent to Line5
    angle_start_arc2 = np.arctan2(Point2[1] - Point1[1], Point2[0] - Point1[0])
    arc2_angles = np.linspace(angle_start_arc2, angle_start_arc2 - 2*np.pi, 1000)
    arc2_coords = []
    arc2_end = None

    for i in range(1, len(arc2_angles)):
        a1 = arc2_angles[i - 1]
        a2 = arc2_angles[i]
        p1 = Point1 + f * np.array([np.cos(a1), np.sin(a1)])
        p2 = Point1 + f * np.array([np.cos(a2), np.sin(a2)])
        tangent_vec = p2 - p1
        tangent_unit = tangent_vec / np.linalg.norm(tangent_vec)
        angle_diff = np.arccos(np.clip(np.dot(tangent_unit, Line_theta_dir), -1.0, 1.0))
        # print(f"Angle diff: {angle_diff:.4f} radians")  # Optional debug

        arc2_coords.append(p2)
        if np.isclose(angle_diff, np.deg2rad(180), atol=1e-2):
            arc2_end = p2
            break

    arc2_coords = np.array(arc2_coords)
    if arc2_end is None:
        arc2_end = arc2_coords[-1]

    # Line5: from arc2_end along Line_theta_dir until it hits y=0
    t_line5 = -arc2_end[1] / Line_theta_dir[1]
    Point3 = arc2_end + t_line5 * Line_theta_dir

    # Arc3 (center at Point0, radius Ro)
    arc3_angles = np.linspace(0, angle2, 100)
    arc3_x = Ro * np.cos(arc3_angles)
    arc3_y = Ro * np.sin(arc3_angles)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Lines
    ax.plot([0, Ro], [0, 0], 'k', label='Line1')
    ax.plot([0, 1.2*Ro*Line2_dir[0]], [0, 1.2*Ro*Line2_dir[1]], 'b--', label='Line2')
    ax.plot([Ri, Point1[0]], [0, Point1[1]], 'r', label='θ/2 Line')
    ax.plot([Point1[0], Point2[0]], [Point1[1], Point2[1]], 'g', label='Line3')
    ax.plot([0, 1.2*Ro*Line4_dir[0]], [0, 1.2*Ro*Line4_dir[1]], 'purple', label='Line4')
    ax.plot([arc2_end[0], Point3[0]], [arc2_end[1], Point3[1]], 'orange', label='Line5 (tangent)')

    # Arcs
    ax.plot(arc1_x, arc1_y, 'c', label='Arc1 (Rp + f)')
    ax.plot(arc2_coords[:, 0], arc2_coords[:, 1], 'm', label='Arc2 (tangent to Line5)')
    ax.plot(arc3_x, arc3_y, 'brown', label='Arc3 (Ro perimeter)')

    # Annotated points
    offset = np.array([0.01, 0.015])  # Increased and consistent offset for better label positioning
    for pt, name in zip([Point0, Point1, Point2, arc2_end, Point3],
                        ['Point0', 'Point1', 'Point2', 'Arc2_End', 'Point3']):
        ax.plot(*pt, 'o')
        ax.annotate(name, pt + offset, fontsize=10)

    # Annotate key parameters
    # w: Web thickness (Ro - Rp - f)
    w_label_x = (arc1_x[-1] + arc3_x[-1]) / 2
    w_label_y = (arc1_y[-1] + arc3_y[-1]) / 2
    ax.annotate("w", xy=(arc1_x[-1], arc1_y[-1]), xytext=(w_label_x + 0.005, w_label_y),
                arrowprops=dict(arrowstyle="->", color='black'), fontsize=12, color='black')

    # Ri: Internal radius
    ax.annotate("Ri", xy=(Ri, 0.0), xytext=(Ri + 0.01, 0.005),
                arrowprops=dict(arrowstyle="->", color='blue'), fontsize=12, color='blue')

    # Ro: External radius
    ax.annotate("Ro", xy=(Ro, 0.0), xytext=(Ro - 0.02, 0.01),
                arrowprops=dict(arrowstyle="->", color='red'), fontsize=12, color='red')

    # f: Fillet radius (annotated at Point2)
    ax.annotate("f", xy=(Point2[0], Point2[1]), xytext=(Point2[0] + 0.01, Point2[1] + 0.01),
                arrowprops=dict(arrowstyle="->", color='green'), fontsize=12, color='green')

    # e: angle coefficient is abstract, could add text info
    ax.text(0.05, 0.95, f"e = {e}", transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    ax.set_title("Arc2 Ends When Tangent to Line5 (Using w)")
    ax.legend()
    plt.show()
    # fig.savefig(f"WORK/geometry_base.png", dpi=300, bbox_inches='tight')

# === SIMPLE GRAIN GEOMETRY PLOTTING FUNCTIONS ===

def plot_bates_geometry(diameter, core_diameter, show_plot=True, save_path=None):
    """
    Plots a simple BATES grain geometry as two concentric circles.
    
    Parameters:
        diameter (float): Outer diameter of the grain [m]
        core_diameter (float): Inner core diameter of the grain [m]
        show_plot (bool): Whether to display the plot
        save_path (str): If provided, saves the plot to this file
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    outer_circle = plt.Circle((0, 0), diameter / 2, fill=False, color='blue', label='Outer Diameter')
    inner_circle = plt.Circle((0, 0), core_diameter / 2, fill=False, color='red', label='Core Diameter')

    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)
    ax.set_xlim(-diameter / 2 * 1.2, diameter / 2 * 1.2)
    ax.set_ylim(-diameter / 2 * 1.2, diameter / 2 * 1.2)
    ax.set_title("BATES Grain Geometry")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_endburner_geometry(diameter, show_plot=True, save_path=None):
    """
    Plots a simple endburner geometry as a single circle.

    Parameters:
        diameter (float): Diameter of the grain [m]
        show_plot (bool): Whether to display the plot
        save_path (str): If provided, saves the plot to this file
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), diameter / 2, fill=False, color='green', label='Outer Diameter')

    ax.add_patch(circle)
    ax.set_xlim(-diameter / 2 * 1.2, diameter / 2 * 1.2)
    ax.set_ylim(-diameter / 2 * 1.2, diameter / 2 * 1.2)
    ax.set_title("Endburner Grain Geometry")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def generate_star_geometry_from_scratch_2(diameter, pointLength, pointWidth, numPoints, show_plot=True, save_path=None):
    """
    Generate and plot a full star geometry from basic parameters: diameter, pointLength (L), pointWidth (W), and numPoints.

    Parameters:
        diameter (float): Outer diameter of the star grain [m]
        pointLength (float): Radial length of each star point [m]
        pointWidth (float): Width of each point at its base [m]
        numPoints (int): Number of points in the star
        show_plot (bool): Whether to display the plot
        save_path (str): If provided, saves the plot to this file
    """
    Ro = diameter / 2
    Ri = Ro - pointLength
    N = int(numPoints)

    base_angle = 2 * np.pi / N
    half_width_angle = np.arcsin(pointWidth / (2 * Ro))

    angles = []
    radii = []

    for i in range(N):
        outer_angle = i * base_angle
        half_w = half_width_angle

        theta_inner1 = outer_angle - half_w
        theta_outer = outer_angle
        theta_inner2 = outer_angle + half_w

        # Star point (triangle spike)
        angles.extend([theta_inner1, theta_outer, theta_inner2])
        radii.extend([Ri, Ro, Ri])

        # Smooth arc from current valley to next valley
        next_theta_inner1 = (outer_angle + base_angle) - half_w
        arc_angles = np.linspace(theta_inner2, next_theta_inner1, 10, endpoint=False)[1:]

        angles.extend(arc_angles)
        radii.extend([Ri] * len(arc_angles))

    # Close the loop
    angles.append(angles[0])
    radii.append(radii[0])

    x = np.array(radii) * np.cos(angles)
    y = np.array(radii) * np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.plot(x, y, '-', label='Star Profile')  # Only line, no markers
    ax.set_title(f"Star Geometry (N={N})")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_geometry_and_curve_from_misclassified(functions_csv_list, params_csv_list, pressure_curve=None, idx=0, validation_csv=None, validation_index=None):
    """
    Given a misclassified curve and its index in the validation CSV, find its corresponding geometry parameters
    and plot both the geometry and the curve side by side using the appropriate plotting function.

    Args:
        functions_csv_list (list of str): List of paths to candidate functions.csv files
        params_csv (str): Path to the single params.csv file (with header)
        pressure_curve (np.ndarray): The misclassified pressure curve (for plotting only)
        idx (int): Index label for the figure
        validation_csv (str): Path to the validation CSV file to extract true unmodified values
        validation_index (int): Index of the misclassified curve in the validation CSV
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from master_thesis.my_functions.helper_functions import (
        generate_star_geometry_from_scratch_2,
        plot_bates_geometry,
        plot_endburner_geometry
    )

    if validation_csv is None or validation_index is None:
        print(f"❌ Must provide validation_csv and validation_index to locate original values.")
        return

    # Read lines manually to support variable-length rows
    with open(validation_csv, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if validation_index >= len(lines):
        print(f"❌ Validation index {validation_index} out of range.")
        return
    print("Index: ", validation_index)
    row = lines[validation_index].split(',')
    key = np.array(row[:5], dtype=float)

    matched_idx = None
    found_class = None

    for class_id, path in enumerate(functions_csv_list):
        try:
            with open(path, 'r') as f:
                lines = [line.strip().split(',') for line in f if line.strip()]
            print(f"✅ Loaded functions.csv for class {class_id} from {path}")
        except Exception as e:
            print(f"❌ Error loading functions.csv for class {class_id} from {path}. Skipping this class.\n{e}")
            continue

        for i, row in enumerate(lines):
            try:
                row_vals = np.array([float(x) for x in row if x != ""], dtype=float)
            except Exception:
                continue
            if len(row_vals) < 5:
                continue
            # print(f"Checking row {i} in class {class_id}: {row_vals[:5]} against key {key}")
            if np.allclose(row_vals[:5], key, rtol=0, atol=1e-8):
                matched_idx = i
                found_class = class_id
                print(f"✅ Found match in class {class_id} at index {matched_idx} for key {key} and row_vals {row_vals[:5]}")
                break
        if matched_idx is not None:
            break

    if matched_idx is None:
        print(f"⚠️ Error: No unique match found for curve #{idx} using validation index {validation_index}.")
        return

    # Load the correct params file for the found class
    params_path = params_csv_list[found_class]
    params_df = pd.read_csv(params_path, engine='python')
    

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # plt.sca(axs[0])
    if found_class == 1:
        try:
            params = params_df.loc[matched_idx]  # header row offsets index naturally
            diameter = float(params["diameter"])
            pointLength = float(params["pointLength"])
            pointWidth = float(params["pointWidth"])
            numPoints = int(round(params["numPoints"], 0))
            print(f"✅ Successfully parsed parameters for class {found_class} at index {matched_idx}: "
                f"diameter={diameter}, pointLength={pointLength}, pointWidth={pointWidth}, numPoints={numPoints}")
        except Exception as e:
            print(f"❌ Error parsing parameters at index {matched_idx}: {e}")

        generate_star_geometry_from_scratch_2(
            diameter, pointLength, pointWidth, numPoints, show_plot=True
        )
        print("Generated plot")
        # axs[0].set_title(f"Star Geometry (Idx {matched_idx})")
    elif found_class == 0:
        try:
            params = params_df.loc[matched_idx]  # header row offsets index naturally
            diameter = float(params["diameter"])
            length = float(params["length"])
            print(f"✅ Successfully parsed parameters for class {found_class} at index {matched_idx}: "
                f"diameter={diameter}, pointLength={length}")
        except Exception as e:
            print(f"❌ Error parsing parameters at index {matched_idx}: {e}")

        plot_bates_geometry(diameter, length, show_plot=False)
        # axs[0].set_title(f"Bates Geometry (Idx {matched_idx})")
    elif found_class == 2:
        try:
            params = params_df.loc[matched_idx]  # header row offsets index naturally
            diameter = float(params["diameter"])
            print(f"✅ Successfully parsed parameters for class {found_class} at index {matched_idx}: "
                f"diameter={diameter}")
        except Exception as e:
            print(f"❌ Error parsing parameters at index {matched_idx}: {e}")

        plot_endburner_geometry(diameter, show_plot=False)
        # axs[0].set_title(f"Endburner Geometry (Idx {matched_idx})")
    # else:
        # axs[0].text(0.5, 0.5, "Unknown Class", ha='center', va='center')
        # axs[0].set_title(f"Unknown Geometry (Idx {matched_idx})")

    if pressure_curve is not None:
        print(f"✅ Plotting pressure curve for misclassified sample {idx} at index {matched_idx}")
        # axs[1].plot(pressure_curve)
        # axs[1].set_title("Pressure Curve")

    # fig.suptitle(f"Misclassified Sample {idx}")
    # plt.tight_layout()
    # plt.show()

    # === REVERSE PREPROCESSING FUNCTION ===
def reverse_preprocessing(normalized_curve):
    """
    Reverses the normalization and padding used in FunctionDataset.
    Assumes padding was added as zeros and normalization was done per curve using min-max scaling.

    Args:
        normalized_curve (np.ndarray): Normalized and padded input

    Returns:
        np.ndarray: Best-effort reconstruction of original curve shape and relative values
    """
    import torch
    # Remove trailing padding (assumed to be 0)
    curve = normalized_curve
    if isinstance(curve, torch.Tensor):
        curve = curve.squeeze().cpu().numpy()
    trimmed = curve.copy()
    while len(trimmed) > 0 and np.isclose(trimmed[-1], 0, atol=1e-6):
        trimmed = trimmed[:-1]

    # Cannot reverse min-max without original min and max, but we can scale back to 0-1
    # This still won't match original data exactly
    return trimmed

# gui_callbacks.py

def callback_update(iterations, total=None):
    try:
        import gui_launcher
        gui_launcher.update_progress_bar(iterations, total)
    except Exception as e:
        print(f"[WARN] Callback update skipped: {e}")
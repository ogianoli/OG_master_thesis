import pygmo as pg
import numpy as np
import sys
import os
import time
from tqdm import tqdm
from sklearn.metrics import r2_score
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from master_thesis.my_tool.GUI_test.helper_functions_CLI import *

iteration_params = []
iteration_errors = []
iteration_curves_pressure = []
iteration_curves_time = []
iteration_success_flags = []

global progress_bar, progress_counter, expected_evaluations
progress_bar = None
progress_counter = 0
expected_evaluations = None  # to be set externally

# Global progress queue for cross-module progress updates
global_progress_queue = None
class OptimizationHistory:
    def __init__(self):
        self.params = []
        self.errors = []
        self.curves_pressure = []
        self.curves_time = []
        self.success_flags = []
        self.best_curves = []
        self.best_errors = []

    def append(self, x, error, pressure, time, success):
        self.params.append(tuple(x))
        self.errors.append(error)
        self.curves_pressure.append(pressure)
        self.curves_time.append(time)
        self.success_flags.append(success)
        # Track best-so-far curve and error for live GUI plotting
        if not self.best_errors or error < min(self.best_errors):
            self.best_curves.append(pressure)
            self.best_errors.append(error)
        else:
            # Repeat previous best
            self.best_curves.append(self.best_curves[-1])
            self.best_errors.append(self.best_errors[-1])


def penalty_throat_greater_than_diameter(throat, diameter, scale=1e4, base_penalty=100.0):
    if throat >= diameter:
        violation = throat - diameter
        return violation * scale + base_penalty
    return None

def penalty_burntime_mismatch(sim_time, target_time, weight=5.0):
    try:
        burntime_diff = abs(sim_time[-1] - target_time)
        return weight * (burntime_diff) ** 2
    except Exception:
        return weight * 100.0

class PressureCurveFitProblem:
    def __init__(self, target_pressure, time_vector, saving_file, motor_file, variable_config, history=None):
        self.variable_config = variable_config
        self.variable_names = [v["name"] for v in variable_config]
        self.lower_bounds = [v["bounds"][0] for v in variable_config]
        self.upper_bounds = [v["bounds"][1] for v in variable_config]
        self.target_pressure = target_pressure
        self.time_vector = time_vector
        self.motor_file = motor_file
        self.saving_file = saving_file
        # For GUI live history
        self.history = history

    def fitness(self, x):
        param_values = {}
        for i, var in enumerate(self.variable_config):
            name = var["name"]
            if var.get("type") == "discrete":
                param_values[name] = int(round(x[i]))
            else:
                param_values[name] = x[i]
        diameter = param_values.get("diameter")
        throat = param_values.get("throat")

        if throat is not None and diameter is not None:
            penalty = penalty_throat_greater_than_diameter(throat, diameter)
            if penalty is not None:
                iteration_params.append(tuple(x))
                iteration_errors.append(penalty)
                iteration_curves_pressure.append([np.nan, np.nan, np.nan])
                iteration_curves_time.append([np.nan, np.nan, np.nan])
                iteration_success_flags.append("no")
                if self.history is not None:
                    self.history.append(x, penalty, [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], "no")
                return [penalty]

        update_motor_file(self.motor_file, **param_values)
        run_openmotor(self.motor_file, self.saving_file)

        try:
            sim_pressure = get_array(self.saving_file, "Chamber Pressure(Pa)")
            sim_time = get_array(self.saving_file, "Time(s)")

            if len(sim_pressure) < 3 or np.any(np.isnan(sim_pressure)) or np.any(sim_pressure <= -1):
                raise ValueError("Invalid pressure data")

            # --- Norm Error ---
            sim_pressure_norm = normalize(sim_pressure, self.target_pressure)
            shape_error = 1 - r2_score(self.target_pressure, sim_pressure_norm)

            # --- Burntime penalty ---
            burntime_penalty = penalty_burntime_mismatch(sim_time, self.time_vector[-1])

            # --- Final error ---
            error = shape_error + burntime_penalty
            
            print(f"[INFO] Shape error: {shape_error:.5f}, Burntime penalty: {burntime_penalty:.5f}, Total error: {error:.5f}")

            sim_success = "yes"
        except Exception as e:
            print(f"[WARNING] Simulation failed at x={x}: {e}")
            sim_pressure = np.array([np.nan, np.nan, np.nan])
            sim_time = np.array([np.nan, np.nan, np.nan])
            error = 1000.0
            sim_success = "no"

        iteration_params.append(tuple(x))
        iteration_errors.append(error)
        iteration_curves_pressure.append(sim_pressure)
        iteration_curves_time.append(sim_time)
        iteration_success_flags.append(sim_success)
        # Store to history for GUI
        if self.history is not None:
            self.history.append(x, error, sim_pressure, sim_time, sim_success)

        global progress_counter, expected_evaluations, progress_bar
        if progress_bar is None and expected_evaluations:
            progress_bar = tqdm(total=expected_evaluations, ncols=90)

        progress_counter += 1
        # Use global_progress_queue if available
        from master_thesis.my_tool.GUI_test.optimizer_GUI import global_progress_queue
        if global_progress_queue:
            global_progress_queue.put({
                "progress": (progress_counter / expected_evaluations) * 100 if expected_evaluations else 0,
                "eta": f"Iter {progress_counter}/{expected_evaluations}"
            })
        if progress_bar:
            # Abbreviations for compact display
            short_names = {"diameter": "do", "throat": "th", "length": "l", "coreDiameter": "dc"}
            desc = ', '.join([f"{short_names.get(name, name[:2])}: {val:.4f}"
                            for name, val in zip(self.variable_names, x)])
            progress_bar.set_description(f"🛠 {desc}")
            progress_bar.update(1)
            progress_bar.refresh()
            sys.stdout.flush()
        
        print("------------------------")
        print("--------------------------")
        return [error]

    def get_bounds(self):
        return (self.lower_bounds, self.upper_bounds)  # Adjust bounds as needed
# === IMPORTS AND PATH SETUP ===

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import time
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO
from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from master_thesis.my_tool.GUI_doublemotor.helper_functions_GUI import *
from master_thesis.my_tool.GUI_doublemotor.optimizer_GUI import (
    PressureCurveFitProblem, iteration_params, iteration_errors,
    iteration_curves_pressure, iteration_curves_time, progress_bar, expected_evaluations,
    iteration_success_flags
)
# from master_thesis.my_functions.transformer import ThrustToPressureTransformer
# from master_thesis.openmotor.motorlib.propellant import Propellant
import master_thesis.my_tool.GUI_doublemotor.optimizer_GUI as optimizer


# === CONFIGURATION SECTION ===

def run_optimization(
    target_file,
    motor_file,
    output_dir,
    pop_size=25,
    generations=100,
    live_history=False,
    progress_queue=None
):
    history = [] if live_history else None
    # === VARIABLE CONFIGURATION ===
    optimization_variables = [
        {
            "name": "diameter",
            "bounds": (0.026, 0.1625),
            "label": "Diameter Outer (m)",
            "color": "teal",
            "type": "continuous"
        },
        {
            "name": "length",
            "bounds": (0.08, 0.5),
            "label": "Length Grain (m)",
            "color": "indigo",
            "type": "continuous"
        },
        {
            "name": "pointLength",
            "bounds": (0.006, 0.0375),
            "label": "Point Length (m)",
            "color": "darkorange",
            "type": "continuous"
        },
        {
            "name": "pointWidth",
            "bounds": (0.0032, 0.02),
            "label": "Point Width (m)",
            "color": "green",
            "type": "continuous"
        },
        {
            "name": "numPoints",
            "bounds": (4, 10),
            "label": "Number of Points",
            "color": "blue",
            "type": "discrete"
        }
    ]

    # === Check if all optimization variables exist in motor file ===
    print(f"[DEBUG] Checking variables")
    check_variables(optimization_variables, motor_file)
    print(f"[DEBUG] POINT")

    # === SETUP OUTPUT DIRECTORY ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_num = 1
    data_dir = output_dir
    metadata_csv_path = f"{data_dir}/optimization_metadata{data_num}.csv"
    pressure_curves_csv_path = f"{data_dir}/optimization_curves{data_num}.csv"
    times_curves_csv_path = f"{data_dir}/optimization_curves_time{data_num}.csv"
    plot_dir = f"{data_dir}/plot{data_num}.png"
    plot_dir_zoomed = f"{data_dir}/plot_zoomed{data_num}.png"
    saving_file = f"{data_dir}/raw_data.csv"
    
    # === LOAD TARGET PRESSURE DATA ===
    target_pressure = get_array(target_file, "Chamber Pressure(Pa)")
    time_vector = get_array(target_file, "Time(s)")

    expected_evaluations = pop_size * (generations + 1)
    optimizer.expected_evaluations = expected_evaluations
    start_time = time.time()
    problem = PressureCurveFitProblem(
        target_pressure, time_vector, saving_file, motor_file, optimization_variables
    )
    if progress_queue:
        import optimizer_GUI
        optimizer_GUI.global_progress_queue = progress_queue
        optimizer.progress_queue = progress_queue
    prob = pg.problem(problem)
    algo = pg.algorithm(pg.cmaes(gen=generations, sigma0=1.0, force_bounds=True))
    algo.set_verbosity(1)
    pop = pg.population(prob, size=pop_size)
    pop = algo.evolve(pop)

    # === RESULTS ===
    best_x = pop.champion_x
    best_f = pop.champion_f
    if progress_bar:
        progress_bar.close()
    else:
        print("Progress bar not initialized..")
    print("Optimization completed.")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    print("\nBest variables found:", best_x)
    print("Error:", best_f[0])

    print("[DEBUG_00]  HISTORY SAVED.")

    # === SAVE HISTORY TO DISK ===
    print("Saving optimization history to CSV...")
    save_metadata(
        iteration_params, iteration_errors, iteration_success_flags, metadata_csv_path, optimization_variables
    )
    save_pressure_curves(
        iteration_curves_pressure, pressure_curves_csv_path
    )
    # save_time_vectors(iteration_curves_time, times_curves_csv_path)
    save_run_info(data_dir, data_num, time_vector, motor_file, best_x, best_f)

    # === PLOTTING ===
    # final_sim_time = get_array(saving_file, "Time(s)")
    # final_sim_pressure = get_array(saving_file, "Chamber Pressure(Pa)")
    final_sim_time, final_sim_pressure = get_best_curve(metadata_csv_path, pressure_curves_csv_path, best_x, time_step=0.1)
    # if live_history:
    #     # Append pressure and time for each available iteration if available
    #     for t_curve, p_curve in zip(iteration_curves_time, iteration_curves_pressure):
    #         if t_curve is not None and p_curve is not None:
    #             history.append((t_curve.tolist(), p_curve.tolist()))
    plot_all(
        iteration_curves_pressure, iteration_curves_time, iteration_errors,
        iteration_params, time_vector, target_pressure,
        final_sim_time, final_sim_pressure, best_x, best_f,
        pop_size, plot_dir, optimization_variables
    )
    plot_all(
        iteration_curves_pressure, iteration_curves_time, iteration_errors,
        iteration_params, time_vector, target_pressure,
        final_sim_time, final_sim_pressure, best_x, best_f,
        pop_size, plot_dir_zoomed, optimization_variables, gen_range=(10, generations+1)
    )
    # plot_all(
    #     iteration_curves_pressure, iteration_curves_time, iteration_errors,
    #     iteration_params, time_vector, target_pressure,
    #     best_x, best_f,
    #     pop_size, plot_dir_zoomed, optimization_variables, metadata_csv_path, pressure_curves_csv_path
    # )

    # plot_all(
    #     iteration_curves_pressure, iteration_curves_time, iteration_errors,
    #     iteration_params, time_vector, target_pressure,
    #     best_x, best_f,
    #     pop_size, plot_dir_zoomed, optimization_variables, metadata_csv_path, pressure_curves_csv_path, gen_range=(10, generations + 1)
    # )
    
    return {
        "best_x": best_x,
        "best_f": best_f,
        "plot_dir": plot_dir,
        "plot_dir_zoomed": plot_dir_zoomed,
        "metadata_csv_path": metadata_csv_path,
        "pressure_curves_csv_path": pressure_curves_csv_path,
        "output_dir": output_dir,
        "history": history,
        "done": True
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Star optimization.")
    parser.add_argument("target_file", help="Path to CSV file with target curve")
    parser.add_argument("motor_file", help="Path to .ric motor file")
    parser.add_argument("output_dir", help="Directory to save results")
    parser.add_argument("--pop_size", type=int, default=25, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    args = parser.parse_args()
    run_optimization(
        args.target_file,
        args.motor_file,
        args.output_dir,
        pop_size=args.pop_size,
        generations=args.generations
    )
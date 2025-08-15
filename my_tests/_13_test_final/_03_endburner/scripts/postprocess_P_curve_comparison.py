import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
from master_thesis.my_tool.GUI_doublemotor.helper_functions_GUI import *



def postprocess_opt():
    opt_num = input("Enter the optimization number (e.g., 27 for opt27): ").strip()
    test_dir = "_13_test_final/_03_endburner"
    opt_dir = f"master_thesis/my_tests/{test_dir}/data/opt{opt_num}"

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
        }
        # ,{
        #     "name": "throat",
        #     "bounds": (0.001, 0.01),
        #     "label": "Throat diameter (m)",
        #     "color": "blue",
        #     "type": "continuous"
        # }
    ]

        # Check for subdirectories
    subdirs = [name for name in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, name))]
    if subdirs:
        print("Available subdirectories:")
        for i, sub in enumerate(subdirs):
            print(f"{i + 1}. {sub}")
        selected_idx = int(input("Select subdirectory index: ")) - 1
        selected_subdir = subdirs[selected_idx]
        opt_dir = os.path.join(opt_dir, selected_subdir)
    info_file = os.path.join(opt_dir, f"info1.txt")
    target_file = (f"/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/{test_dir}/results/target.csv")

    if not os.path.exists(info_file):
        print(f"❌ Info file not found: {info_file}")
        return

    # Parse info
    with open(info_file, 'r') as f:
        lines = f.readlines()
    
    time_step = float(lines[0].split(":")[-1].strip())
    motor_file = lines[1].split(":")[-1].strip()
    motor_file = os.path.join(os.path.dirname(motor_file), "best_motor.ric")    # motor_file = (f"/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/{test_dir}/motor_target.ric")
    print("motor_file: ", motor_file)
    comment = lines[2].split(":")[-1].strip()
    best_x_str = lines[3].split(":")[-1].strip()
    best_x = tuple(map(float, best_x_str.split(",")))
    print("best_x: ", best_x)
    best_f = float(lines[4].split(":")[-1].strip())

    # Update motor file and rerun simulation with best_x
    saving_file = os.path.join(f"/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/{test_dir}/postprocessing/postprocess_sim{opt_num}.csv")
    print("saving_file: ", saving_file)
    variable_names = [var["name"] for var in optimization_variables]
    # variable_names = ["diameter", "length", "pointLength", "pointWidth", "numPoints"]  # Update this list according to your current optimization_variables
    param_values = dict(zip(variable_names, best_x))
    update_motor_file(motor_file, **param_values)
    # update_motor_file(motor_file, *best_x)
    run_openmotor(motor_file, saving_file)

    # Get simulation and target data
    sim_pressure = get_array(saving_file, "Chamber Pressure(Pa)")
    sim_time = get_array(saving_file, "Time(s)")
    target_pressure = get_array(target_file, "Chamber Pressure(Pa)")
    target_time = get_array(target_file, "Time(s)")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(sim_time, sim_pressure, label="Simulated Pressure", color='blue')
    plt.plot(target_time, target_pressure, label="Target Pressure", linestyle='--', color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title(f"Target vs Simulation Pressure Curve Comparison (opt{opt_num})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(opt_dir, f"pressure_curve_comparison_{opt_num}.png"))
    plt.show()


if __name__ == "__main__":
    postprocess_opt()
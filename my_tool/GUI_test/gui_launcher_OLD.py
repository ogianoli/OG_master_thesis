import queue
progress_queue = queue.Queue()
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import os
import tempfile
import time
from PIL import Image, ImageTk
import re
def get_next_run_directory(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing = [d for d in os.listdir(base_path) if d.startswith("run_")]
    run_numbers = [int(re.findall(r'\d+', d)[0]) for d in existing if re.findall(r'\d+', d)]
    next_number = max(run_numbers, default=0) + 1
    return os.path.join(base_path, f"run_{next_number:03d}")

# Import classifier and optimization functions
from master_thesis.my_tool.GUI_test.predict_class_from_file_GUI import predict_class_from_file
# Removed run_pipeline import since it's CLI-based and unused in GUI
from the_main_bates_GUI import run_optimization as run_bates
from the_main_star_GUI import run_optimization as run_star
from the_main_endburner_GUI import run_optimization as run_endburner
from master_thesis.my_functions.helper_functions import (
    plot_bates_geometry, plot_endburner_geometry, generate_star_geometry_from_scratch_2
)
from optimizer_GUI import iteration_curves_time, iteration_curves_pressure

root = tk.Tk()
root.title("SRM Classifier + Optimizer")
root.geometry("800x600")

# --- Notebook (Tabs) ---
notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=3, rowspan=30, sticky="nsew")

main_tab = tk.Frame(notebook)
results_tab = tk.Frame(notebook)

notebook.add(main_tab, text="Main")
notebook.add(results_tab, text="Results")

csv_path_var = tk.StringVar()
thrust_var = tk.StringVar()
burntime_var = tk.StringVar()
classification_result = tk.StringVar()
allowed_bates = tk.BooleanVar(value=True)
allowed_star = tk.BooleanVar(value=True)
allowed_endburner = tk.BooleanVar(value=True)

tk.Label(main_tab, text="Allow Classes:").grid(row=3, column=0, sticky="w")
tk.Checkbutton(main_tab, text="Bates", variable=allowed_bates).grid(row=4, column=0, sticky="w")
tk.Checkbutton(main_tab, text="Star", variable=allowed_star).grid(row=5, column=0, sticky="w")
tk.Checkbutton(main_tab, text="Endburner", variable=allowed_endburner).grid(row=6, column=0, sticky="w")
status_var = tk.StringVar()
pop_size_var = tk.StringVar(value="20")
generations_var = tk.StringVar(value="30")

cancel_flag = threading.Event()

# --- Zoomed image label ---
zoomed_image_label = None

# --- Matplotlib Figure ---
fig, ax = plt.subplots(figsize=(6, 3))
canvas = FigureCanvasTkAgg(fig, master=main_tab)
canvas.get_tk_widget().grid(row=9, column=0, columnspan=4, pady=10)

# --- Geometry Figure (empty placeholder) ---
geometry_fig = None
geometry_canvas = None
geometry_label = None

# --- Input Fields ---
tk.Label(main_tab, text="CSV Path:").grid(row=0, column=0, sticky="e")
tk.Entry(main_tab, textvariable=csv_path_var, width=50).grid(row=0, column=1, columnspan=2, sticky="we")
def browse_csv():
    path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*")])
    if path:
        csv_path_var.set(path)
tk.Button(main_tab, text="Browse", command=browse_csv).grid(row=0, column=3, sticky="w")

tk.Label(main_tab, text="Or enter thrust (N):").grid(row=1, column=0, sticky="e")
tk.Entry(main_tab, textvariable=thrust_var).grid(row=1, column=1, sticky="we")

tk.Label(main_tab, text="Burntime (s):").grid(row=2, column=0, sticky="e")
tk.Entry(main_tab, textvariable=burntime_var).grid(row=2, column=1, sticky="we")

tk.Label(main_tab, text="Generations:").grid(row=2, column=2, sticky="e")
tk.Entry(main_tab, textvariable=generations_var, width=5).grid(row=2, column=3, sticky="w")

tk.Label(main_tab, text="Population Size:").grid(row=3, column=2, sticky="e")
tk.Entry(main_tab, textvariable=pop_size_var, width=5).grid(row=3, column=3, sticky="w")

tk.Label(main_tab, text="Classification:").grid(row=4, column=1, sticky="e")
tk.Label(main_tab, textvariable=classification_result, font=("Arial",12,"bold")).grid(row=3, column=2, sticky="w")

 # --- Progress label and bar below classification area ---
tk.Label(main_tab, text="Opt. progress:").grid(row=5, column=1, sticky="e")
progress = ttk.Progressbar(main_tab, orient="horizontal", length=200, mode="determinate")
progress.grid(row=5, column=2, columnspan=2, pady=5)
eta_label = tk.Label(main_tab, text="")
eta_label.grid(row=5, column=3, sticky="w")

tk.Label(main_tab, text="Output:").grid(row=6, column=1, sticky="e")
tk.Label(main_tab, textvariable=status_var, fg="red").grid(row=6, column=2, columnspan=4, sticky="w")

# --- Helper: Generate synthetic target curve ---
def generate_target_curve(thrust, burntime, steps=100):
    t = np.linspace(0, float(burntime), steps)
    y = np.ones_like(t) * float(thrust)
    return t, y

# --- Helper: Save curve to CSV for classifier ---
def save_curve_to_csv(t, y):
    path = os.path.join(tempfile.gettempdir(), f"target_{int(time.time())}.csv")
    with open(path, 'w') as f:
        f.write("Time(s),Chamber Pressure(Pa)\n")
        for t_val, y_val in zip(t, y):
            f.write(f"{t_val},{y_val}\n")
    return path
    

# --- Helper: Update Progress bar ---
def update_progress_bar(iterations, total=None):
    if isinstance(progress, ttk.Progressbar):
        progress["value"] = int((iterations / total) * 100) if total else iterations
        eta_label.config(text=f"{int((total - iterations))} iters left" if total else "")
        root.update_idletasks()

# --- Plotting target curve ---
def plot_curve(t, y, sim_time=None, sim_curve=None, label="Target Curve"):
    ax.clear()
    ax.plot(t, y, label=label, color="orange", lw=2)
    if sim_time is not None and sim_curve is not None:
        ax.plot(sim_time, sim_curve, label="Simulated Curve", color="blue", lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure/Thrust")
    ax.set_title("Target & Simulated Curve")
    ax.legend()
    canvas.draw()

# --- Plot geometry ---
def plot_geometry(class_id, best_params):
    global geometry_fig, geometry_canvas, geometry_label
    # Destroy previous geometry plot if any
    if geometry_canvas:
        geometry_canvas.get_tk_widget().destroy()
        geometry_canvas = None
    if geometry_label:
        geometry_label.destroy()
        geometry_label = None
    geometry_fig = plt.Figure(figsize=(3,3))
    geometry_ax = geometry_fig.add_subplot(111)
    # Plot based on class
    if class_id == 0:
        # Bates: diameter, throat, length, coreDiameter
        diameter = best_params[0]
        core_diameter = best_params[3]
        plot_bates_geometry(diameter, core_diameter, show_plot=False, save_path=None)
        theta = np.linspace(0, 2*np.pi, 100)
        geometry_ax.plot(np.cos(theta)*diameter/2, np.sin(theta)*diameter/2, label="Outer")
        geometry_ax.plot(np.cos(theta)*core_diameter/2, np.sin(theta)*core_diameter/2, label="Core")
        geometry_ax.set_aspect('equal')
        geometry_ax.set_title("Bates Geometry")
        geometry_ax.legend()
    elif class_id == 1:
        diameter = best_params[0]
        pointLength = best_params[2]
        pointWidth = best_params[3]
        numPoints = int(round(best_params[4]))
        generate_star_geometry_from_scratch_2(diameter, pointLength, pointWidth, numPoints, show_plot=False, save_path=None)
        Ro = diameter / 2
        Ri = Ro - pointLength
        N = numPoints
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
            angles.extend([theta_inner1, theta_outer, theta_inner2])
            radii.extend([Ri, Ro, Ri])
            next_theta_inner1 = (outer_angle + base_angle) - half_w
            arc_angles = np.linspace(theta_inner2, next_theta_inner1, 10, endpoint=False)[1:]
            angles.extend(arc_angles)
            radii.extend([Ri] * len(arc_angles))
        angles.append(angles[0])
        radii.append(radii[0])
        x = np.array(radii) * np.cos(angles)
        y = np.array(radii) * np.sin(angles)
        geometry_ax.plot(x, y, '-', label='Star Profile')
        geometry_ax.set_aspect('equal')
        geometry_ax.set_title("Star Geometry")
        geometry_ax.legend()
    elif class_id == 2:
        diameter = best_params[0]
        plot_endburner_geometry(diameter, show_plot=False, save_path=None)
        theta = np.linspace(0, 2*np.pi, 100)
        geometry_ax.plot(np.cos(theta)*diameter/2, np.sin(theta)*diameter/2, label="Outer")
        geometry_ax.set_aspect('equal')
        geometry_ax.set_title("Endburner Geometry")
        geometry_ax.legend()
    else:
        geometry_ax.text(0.5, 0.5, "Unknown geometry", ha="center", va="center")
    geometry_canvas = FigureCanvasTkAgg(geometry_fig, master=results_tab)
    geometry_canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

# --- Export buttons ---
def save_curve_plot():
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        fig.savefig(path)

def save_geometry_plot():
    if geometry_fig:
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            geometry_fig.savefig(path)

def save_results_csv(sim_time, sim_curve):
    path = filedialog.asksaveasfilename(defaultextension=".csv")
    if path:
        np.savetxt(path, np.column_stack((sim_time, sim_curve)), delimiter=",", header="Time(s),Chamber Pressure(Pa)", comments='')

tk.Button(main_tab, text="Save Curve Plot", command=save_curve_plot).grid(row=10, column=0)
tk.Button(main_tab, text="Save Geometry Plot", command=save_geometry_plot).grid(row=10, column=1)

# --- Main workflow ---
def run_all():
    # Step 1: get input
    status_var.set("Preparing input...")
    root.update_idletasks()
    t = None
    y = None
    csv_path = None
    if csv_path_var.get():
        csv_path = csv_path_var.get()
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "Chamber Pressure(Pa)" not in df.columns or "Time(s)" not in df.columns:
                messagebox.showerror("CSV Error", "CSV must contain 'Time(s)' and 'Chamber Pressure(Pa)' columns.")
                return
            t = df["Time(s)"].values
            y = df["Chamber Pressure(Pa)"].values
        except Exception as e:
            messagebox.showerror("CSV Error", f"Could not read CSV: {e}")
            return
    elif thrust_var.get() and burntime_var.get():
        try:
            t, y = generate_target_curve(thrust_var.get(), burntime_var.get())
            csv_path = save_curve_to_csv(t, y)
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid thrust or burntime: {e}")
            return
    else:
        messagebox.showerror("Input Error", "Provide either a CSV or thrust + burntime.")
        return
    plot_curve(t, y, label="Target Curve")
    status_var.set("Classifying target curve...")
    root.update_idletasks()

    # Step 2: classify
    try:
        pred_class, probs = predict_class_from_file(csv_path)

        allowed = []
        if allowed_bates.get(): allowed.append(0)
        if allowed_star.get(): allowed.append(1)
        if allowed_endburner.get(): allowed.append(2)

        class_id = None
        for cid, _ in probs:
            if cid in allowed:
                class_id = cid
                break

        if class_id is None:
            messagebox.showerror("Classification", "No allowed class matches the prediction.")
            return
    except Exception as e:
        messagebox.showerror("Classifier Error", f"Failed to classify: {e}")
        return
    class_map = {0: "Bates", 1: "Star", 2: "Endburner"}
    classification_result.set(f"{class_id} = {class_map.get(class_id, 'Unknown')}")
    status_var.set(f"Classified as: {class_map.get(class_id, 'Unknown')} (id={class_id})")
    root.update_idletasks()

    # Step 3: run optimization (in thread)
    def optimization_thread():
        import pandas as pd
        from datetime import datetime
        try:
            status_var.set("Starting optimization...")
            root.update_idletasks()
            motor_files = {
                0: "master_thesis/my_tool/GUI_test/motors/motor_bates.ric",
                1: "master_thesis/my_tool/GUI_test/motors/motor_star.ric",
                2: "master_thesis/my_tool/GUI_test/motors/motor_endburner.ric",
            }
            base_data_dir = os.path.join("master_thesis", "my_tool", "GUI_test", "data")
            output_dir = get_next_run_directory(base_data_dir)
            motor_file = motor_files.get(class_id)
            if not motor_file or not os.path.exists(motor_file):
                # Try to find a .ric file in current dir as fallback
                for f in os.listdir('.'):
                    if f.endswith('.ric'):
                        motor_file = f
                        break
            if not motor_file or not os.path.exists(motor_file):
                messagebox.showerror("Motor File Error", f"No suitable .ric motor file found for class {class_id}. Please add template .ric files to 'motors/' or current directory.")
                status_var.set("Motor file not found.")
                return
            # Choose appropriate optimizer
            if class_id == 0:
                run_func = run_bates
            elif class_id == 1:
                run_func = run_star
            elif class_id == 2:
                run_func = run_endburner
            else:
                messagebox.showerror("Unknown Class", f"Unknown class id {class_id}")
                return
            # Set global progress queue in optimizer_GUI
            import optimizer_GUI
            optimizer_GUI.global_progress_queue = progress_queue
            # Run optimizer
            status_var.set("Optimizing... This may take a while.")
            root.update_idletasks()
            start_time = datetime.now()
            pop_size = int(pop_size_var.get())
            generations = int(generations_var.get())
            result = run_func(csv_path, motor_file, output_dir, pop_size, generations, live_history=True, progress_queue=progress_queue)
            sim_path = os.path.join(output_dir, "raw_data.csv")
            if not os.path.exists(sim_path):
                status_var.set("Simulation output not found.")
                return
            for gen in range(generations):
                if cancel_flag.is_set():
                    status_var.set("Optimization cancelled.")
                    return
                time.sleep(1)  # simulate waiting for generation
                print("debug: generation", gen)
                if len(iteration_curves_time) > gen and len(iteration_curves_pressure) > gen:
                    sim_time = iteration_curves_time[gen]
                    sim_curve = iteration_curves_pressure[gen]
                    plot_curve(t, y, sim_time, sim_curve)
                    progress["value"] = int((gen + 1) / generations * 100)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta = elapsed / (gen + 1) * (generations - gen - 1)
                    eta_label.config(text=f"ETA: {int(eta)}s")
                    root.update_idletasks()
            # Load final simulated curve for overlay
            df = pd.read_csv(sim_path)
            sim_time = df["Time(s)"].values
            sim_curve = df["Chamber Pressure(Pa)"].values
            plot_curve(t, y, sim_time, sim_curve)
            status_var.set("Optimization complete. You can now export results.")
            # Show geometry
            import numpy as np
            plot_geometry(class_id, np.array(result["best_x"]))
            # Show zoomed plot if it exists
            global zoomed_image_label
            if zoomed_image_label:
                zoomed_image_label.destroy()
                zoomed_image_label = None
            zoomed_path = os.path.join(output_dir, "plot_zoomed1.png")
            if os.path.exists(zoomed_path):
                img = Image.open(zoomed_path)
                img = img.resize((400, 300))  # Resize for GUI
                img_tk = ImageTk.PhotoImage(img)
                zoomed_image_label = tk.Label(results_tab, image=img_tk)
                zoomed_image_label.image = img_tk  # Keep reference
                zoomed_image_label.grid(row=0, column=0, padx=10, pady=10)
            save_results_csv(sim_time, sim_curve)
        except Exception as e:
            status_var.set("Optimization failed.")
            messagebox.showerror("Optimization Error", str(e))
    cancel_flag.clear()
    threading.Thread(target=optimization_thread).start()

tk.Button(main_tab, text="Run Opt.", command=run_all, bg="lightgreen", font=("Arial",12,"bold")).grid(row=4, column=2, columnspan=4, pady=10, sticky="e")

def cancel_optimization():
    cancel_flag.set()
    status_var.set("Cancelling...")

tk.Button(main_tab, text="Cancel", command=cancel_optimization, bg="red", font=("Arial", 12)).grid(row=11, column=2, columnspan=4, pady=5, sticky="e")

def poll_progress_queue():
    try:
        while True:
            msg = progress_queue.get_nowait()
            if isinstance(msg, dict):
                progress["value"] = msg.get("progress", 0)
                eta_label.config(text=msg.get("eta", ""))
    except queue.Empty:
        pass
    root.after(500, poll_progress_queue)

poll_progress_queue()

root.mainloop()
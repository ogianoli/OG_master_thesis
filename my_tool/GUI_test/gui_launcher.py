# --- PyQt5-based GUI replacing Tkinter ---
import sys
import os
import numpy as np
import tempfile
import time
import threading
import queue
import re
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFileDialog,
    QLabel, QLineEdit, QPushButton, QCheckBox, QProgressBar, QMessageBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

# Import classifier and optimization functions

from master_thesis.my_tool.GUI_test.predict_class_from_file_GUI import predict_class_from_file
from the_main_bates_GUI import run_optimization as run_bates
from the_main_star_GUI import run_optimization as run_star
from the_main_endburner_GUI import run_optimization as run_endburner
from master_thesis.my_tool.GUI_test.helper_functions_GUI import (
    plot_bates_geometry, plot_endburner_geometry, generate_star_geometry_from_scratch_2
)
from optimizer_GUI import iteration_curves_time, iteration_curves_pressure

progress_queue = queue.Queue()

def get_next_run_directory(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing = [d for d in os.listdir(base_path) if d.startswith("run_")]
    run_numbers = [int(re.findall(r'\d+', d)[0]) for d in existing if re.findall(r'\d+', d)]
    next_number = max(run_numbers, default=0) + 1
    return os.path.join(base_path, f"run_{next_number:03d}")

class SRMGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRM Classifier + Optimizer")
        self.setGeometry(100, 100, 950, 700)
        self.cancel_flag = threading.Event()
        self.progress_queue = progress_queue
        # Main widget and layout
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.main_tab = QWidget()
        self.results_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.results_tab, "Results")
        # Variables
        self.csv_path = ""
        self.thrust = ""
        self.burntime = ""
        self.classification_result = ""
        self.allowed_bates = True
        self.allowed_star = True
        self.allowed_endburner = True
        self.status = ""
        self.pop_size = "25"
        self.generations = "50"
        self.sim_time = None
        self.sim_curve = None
        self.t = None
        self.y = None
        self.class_id = None
        self.best_params = None
        self.param_table = None
        self.output_dir = None
        self.geometry_fig = None
        self.geometry_canvas = None
        self.zoomed_label = None
        self.figure = plt.figure(figsize=(6, 3))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.progress_bar = QProgressBar()
        self.eta_label = QLabel("")
        self.status_label = QLabel("")
        self.class_label = QLabel("")
        self._setup_main_tab()
        self._setup_results_tab()
        self._setup_timer()
        # --- Make this GUI instance available globally for callbacks ---
        # (Assignment moved to main() after instance creation)

    def _setup_main_tab(self):
        layout = QGridLayout()
        # CSV path
        layout.addWidget(QLabel("CSV Path:"), 0, 0, Qt.AlignRight)
        self.csv_path_entry = QLineEdit()
        layout.addWidget(self.csv_path_entry, 0, 1, 1, 2)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_csv)
        layout.addWidget(browse_btn, 0, 3)
        # Thrust/Burntime
        layout.addWidget(QLabel("Or enter thrust (N):"), 1, 0, Qt.AlignRight)
        self.thrust_entry = QLineEdit()
        layout.addWidget(self.thrust_entry, 1, 1)
        layout.addWidget(QLabel("Burntime (s):"), 2, 0, Qt.AlignRight)
        self.burntime_entry = QLineEdit()
        layout.addWidget(self.burntime_entry, 2, 1)
        # Generations/Pop size
        layout.addWidget(QLabel("Generations:"), 2, 2, Qt.AlignRight)
        self.generations_entry = QLineEdit(self.generations)
        layout.addWidget(self.generations_entry, 2, 3)
        layout.addWidget(QLabel("Population Size:"), 3, 2, Qt.AlignRight)
        self.pop_size_entry = QLineEdit(self.pop_size)
        layout.addWidget(self.pop_size_entry, 3, 3)
        # Classes
        layout.addWidget(QLabel("Allow Classes:"), 3, 0)
        self.bates_cb = QCheckBox("Bates")
        self.bates_cb.setChecked(True)
        layout.addWidget(self.bates_cb, 4, 0)
        self.star_cb = QCheckBox("Star")
        self.star_cb.setChecked(True)
        layout.addWidget(self.star_cb, 5, 0)
        self.endburner_cb = QCheckBox("Endburner")
        self.endburner_cb.setChecked(True)
        layout.addWidget(self.endburner_cb, 6, 0)
        # Classification result
        layout.addWidget(QLabel("Classification:"), 4, 1, Qt.AlignRight)
        self.class_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        layout.addWidget(self.class_label, 4, 2, 1, 2)
        # Progress bar
        layout.addWidget(QLabel("Opt. progress:"), 5, 1, Qt.AlignRight)
        layout.addWidget(self.progress_bar, 5, 2)
        layout.addWidget(self.eta_label, 5, 3)
        # Status
        layout.addWidget(QLabel("Output:"), 6, 1, Qt.AlignRight)
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label, 6, 2, 1, 2)
        # Plot
        layout.addWidget(self.canvas, 9, 0, 1, 4)
        # Export buttons
        curve_btn = QPushButton("Save Curve Plot")
        curve_btn.clicked.connect(self.save_curve_plot)
        layout.addWidget(curve_btn, 10, 0)
        geom_btn = QPushButton("Save Geometry Plot")
        geom_btn.clicked.connect(self.save_geometry_plot)
        layout.addWidget(geom_btn, 10, 1)
        # Run/Cancel
        run_btn = QPushButton("Run Opt.")
        run_btn.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 13pt;")
        run_btn.clicked.connect(self.run_all)
        layout.addWidget(run_btn, 4, 3, 1, 1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: red; font-size: 12pt;")
        cancel_btn.clicked.connect(self.cancel_optimization)
        layout.addWidget(cancel_btn, 10, 3, 1, 1)
        self.main_tab.setLayout(layout)

    def _setup_results_tab(self):
        self.results_layout = QVBoxLayout()
        self.results_tab.setLayout(self.results_layout)
        self.param_table = QLabel()
        self.param_table.setStyleSheet("font-family: Courier; font-size: 14pt; padding: 6px;")
        self.results_layout.addWidget(self.param_table)
        self.param_table.setVisible(True)
        self.param_table.repaint()

    def _setup_timer(self):
        from PyQt5.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_progress_queue)
        self.timer.start(100)  # poll every 100 ms

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tool/GUI_test/target/", "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_path_entry.setText(path)

    def update_progress_bar(self, iterations, total=None):
        if total:
            val = int((iterations / total) * 100)
            self.progress_bar.setValue(val)
            self.eta_label.setText(f"{int((total - iterations))} iterations left")
        else:
            self.progress_bar.setValue(iterations)
            self.eta_label.setText("")
        QApplication.processEvents()

    def plot_curve(self, t, y, sim_time=None, sim_curve=None, label="Target Curve"):
        self.ax.clear()
        self.ax.plot(t, y, label=label, color="orange", lw=2)
        if sim_time is not None and sim_curve is not None:
            self.ax.plot(sim_time, sim_curve, label="Simulated Curve", color="blue", lw=2)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pressure/Thrust")
        self.ax.set_title("Target & Simulated Curve")
        self.ax.legend()
        self.canvas.draw()

    def plot_geometry(self, class_id, best_params):
        
        import os

        save_path = os.path.join(self.output_dir, "geometry_plot.png")

        if class_id == 0:
            diameter = best_params[0]
            core_diameter = best_params[3]
            plot_bates_geometry(diameter, core_diameter, show_plot=False, save_path=save_path)

        elif class_id == 1:
            diameter = best_params[0]
            pointLength = best_params[2]
            pointWidth = best_params[3]
            numPoints = int(round(best_params[4]))
            generate_star_geometry_from_scratch_2(diameter, pointLength, pointWidth, numPoints, show_plot=False, save_path=save_path)

        elif class_id == 2:
            diameter = best_params[0]
            plot_endburner_geometry(diameter, show_plot=False, save_path=save_path)

        else:
            print(f"[ERROR] Unknown class_id: {class_id}. Cannot plot geometry.")

    def save_curve_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Curve Plot", "", "PNG Files (*.png)")
        if path:
            self.figure.savefig(path)

    def save_geometry_plot(self):
        if self.geometry_canvas:
            path, _ = QFileDialog.getSaveFileName(self, "Save Geometry Plot", "", "PNG Files (*.png)")
            if path and self.geometry_canvas:
                self.geometry_canvas.figure.savefig(path)

    def save_results_csv(self, sim_time, sim_curve):
        path, _ = QFileDialog.getSaveFileName(self, "Save Results CSV", "", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, np.column_stack((sim_time, sim_curve)), delimiter=",", header="Time(s),Chamber Pressure(Pa)", comments='')

    def generate_target_curve(self, thrust, burntime, steps=100):
        t = np.linspace(0, float(burntime), steps)
        y = np.ones_like(t) * float(thrust)
        return t, y

    def save_curve_to_csv(self, t, y):
        path = os.path.join(tempfile.gettempdir(), f"target_{int(time.time())}.csv")
        with open(path, 'w') as f:
            f.write("Time(s),Chamber Pressure(Pa)\n")
            for t_val, y_val in zip(t, y):
                f.write(f"{t_val},{y_val}\n")
        return path

    def run_all(self):
        self.status_label.setText("Preparing input...")
        QApplication.processEvents()
        self.t = None
        self.y = None
        csv_path = None
        # Get inputs
        csv_path = self.csv_path_entry.text().strip()
        thrust = self.thrust_entry.text().strip()
        burntime = self.burntime_entry.text().strip()
        self.pop_size = self.pop_size_entry.text().strip()
        self.generations = self.generations_entry.text().strip()
        self.allowed_bates = self.bates_cb.isChecked()
        self.allowed_star = self.star_cb.isChecked()
        self.allowed_endburner = self.endburner_cb.isChecked()
        try:
            if csv_path:
                import pandas as pd
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                if "Chamber Pressure(Pa)" not in df.columns or "Time(s)" not in df.columns:
                    QMessageBox.critical(self, "CSV Error", "CSV must contain 'Time(s)' and 'Chamber Pressure(Pa)' columns.")
                    return
                self.t = df["Time(s)"].values
                self.y = df["Chamber Pressure(Pa)"].values
            elif thrust and burntime:
                self.t, self.y = self.generate_target_curve(thrust, burntime)
                csv_path = self.save_curve_to_csv(self.t, self.y)
            else:
                QMessageBox.critical(self, "Input Error", "Provide either a CSV or thrust + burntime.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
            return
        self.plot_curve(self.t, self.y, label="Target Curve")
        self.status_label.setText("Classifying target curve...")
        QApplication.processEvents()
        # Classification
        try:
            pred_class, probs = predict_class_from_file(csv_path)
            allowed = []
            if self.allowed_bates: allowed.append(0)
            if self.allowed_star: allowed.append(1)
            if self.allowed_endburner: allowed.append(2)
            class_id = None
            for cid, _ in probs:
                if cid in allowed:
                    class_id = cid
                    break
            if class_id is None:
                QMessageBox.critical(self, "Classification", "No allowed class matches the prediction.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Classifier Error", f"Failed to classify: {e}")
            return
        class_map = {0: "Bates", 1: "Star", 2: "Endburner"}
        self.class_label.setText(f"{class_id} = {class_map.get(class_id, 'Unknown')}")
        self.status_label.setText(f"Classified as: {class_map.get(class_id, 'Unknown')} (id={class_id})")
        QApplication.processEvents()
        self.class_id = class_id
        # Optimization thread
        self.cancel_flag.clear()
        threading.Thread(target=self.optimization_thread, args=(csv_path, class_id), daemon=True).start()

    def optimization_thread(self, csv_path, class_id):
        import pandas as pd
        from datetime import datetime
        try:
            self.status_label.setText("Starting optimization...")
            QApplication.processEvents()
            motor_files = {
                0: "master_thesis/my_tool/GUI_test/motors/motor_bates.ric",
                1: "master_thesis/my_tool/GUI_test/motors/motor_star.ric",
                2: "master_thesis/my_tool/GUI_test/motors/motor_endburner.ric",
            }
            base_data_dir = os.path.join("master_thesis", "my_tool", "GUI_test", "data")
            output_dir = get_next_run_directory(base_data_dir)
            self.output_dir = output_dir
            motor_file = motor_files.get(class_id)
            if not motor_file or not os.path.exists(motor_file):
                for f in os.listdir('.'):
                    if f.endswith('.ric'):
                        motor_file = f
                        break
            if not motor_file or not os.path.exists(motor_file):
                self.status_label.setText("Motor file not found.")
                QMessageBox.critical(self, "Motor File Error", f"No suitable .ric motor file found for class {class_id}. Please add template .ric files to 'motors/' or current directory.")
                return
            if class_id == 0:
                run_func = run_bates
            elif class_id == 1:
                run_func = run_star
            elif class_id == 2:
                run_func = run_endburner
            else:
                QMessageBox.critical(self, "Unknown Class", f"Unknown class id {class_id}")
                return
            import optimizer_GUI
            optimizer_GUI.global_progress_queue = self.progress_queue
            self.status_label.setText("Optimizing... This may take a while.")
            QApplication.processEvents()
            start_time = datetime.now()
            pop_size = int(self.pop_size)
            generations = int(self.generations)
            result = run_func(csv_path, motor_file, output_dir, pop_size, generations, live_history=True, progress_queue=self.progress_queue)
            sim_path = os.path.join(output_dir, "raw_data.csv")
            if not os.path.exists(sim_path):
                self.status_label.setText("Simulation output not found.")
                return
            for gen in range(generations):
                if self.cancel_flag.is_set():
                    self.status_label.setText("Optimization cancelled.")
                    return
                time.sleep(1)
                if len(iteration_curves_time) > gen and len(iteration_curves_pressure) > gen:
                    sim_time = iteration_curves_time[gen]
                    sim_curve = iteration_curves_pressure[gen]
                    self.plot_curve(self.t, self.y, sim_time, sim_curve)
                    self.progress_bar.setValue(int((gen + 1) / generations * 100))
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta = elapsed / (gen + 1) * (generations - gen - 1)
                    self.eta_label.setText(f"ETA: {int(eta)}s")
                    QApplication.processEvents()
            # Final simulated curve
            df = pd.read_csv(sim_path)
            sim_time = df["Time(s)"].values
            sim_curve = df["Chamber Pressure(Pa)"].values
            self.sim_time = sim_time
            self.sim_curve = sim_curve
            self.plot_curve(self.t, self.y, sim_time, sim_curve)
            self.status_label.setText("Optimization complete. You can now export results.")
            import numpy as np
            self.best_params = np.array(result["best_x"])
            print(f"[DEBUG] Displaying params")
            # Display best parameters in results tab
            # param_labels = result.get("optimization_variables", [])

            metadata_path = result.get("metadata_csv_path")
            param_labels = []
            if metadata_path and os.path.exists(metadata_path):
                try:
                    df = pd.read_csv(metadata_path)
                    all_columns = df.columns.tolist()
                    param_labels = all_columns[3:]  # Skip Iteration, Error, Sim_Success
                    print(f"[DEBUG] Extracted param labels from metadata: {param_labels}")
                except Exception as e:
                    print(f"[WARN] Failed to parse metadata for labels: {e}")
            result["optimization_variables"] = param_labels
            if param_labels and isinstance(param_labels[0], dict):
                param_labels = [p['label'] for p in param_labels]
            param_lines = [f"{label:<20}: {val:.4f}" for label, val in zip(param_labels, self.best_params)]
            # --- BEGIN NEW BURN TIME LOGIC BLOCK ---
            burntime = "Unknown"
            try:
                metadata_path = result.get("metadata_csv_path")
                curves_path = result.get("pressure_curves_csv_path")
                info_path = os.path.join(output_dir, "info.txt")
                best_params = np.array(result.get("best_x", []))

                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        for line in f:
                            if line.startswith("time_step:"):
                                time_step = float(line.split(":")[-1].strip())
                                break
                        else:
                            time_step = 1.0
                else:
                    time_step = 1.0
                print("metadata path:", metadata_path)
                if os.path.exists(metadata_path) and os.path.exists(curves_path) and best_params.size > 0:
                    print(f"[DEBUG] Reading metadata from {metadata_path} and curves from {curves_path}")
                    df_meta = pd.read_csv(str(metadata_path))
                    print("df_meta",df_meta)
                    # print(f"[DEBUG] Reading curves from type: {(curves_path.type)} or {type(curves_path)}")
                    # Read curves CSV robustly to handle non-uniform row lengths
                    with open(str(curves_path), 'r') as f:
                        lines = f.readlines()
                    header = lines[0].strip().split(',')
                    data_dict = {key: [] for key in header}
                    for line in lines[1:]:
                        entries = line.strip().split(',')
                        for i in range(len(entries)):
                            key = header[i]
                            try:
                                data_dict[key].append(float(entries[i]))
                            except ValueError:
                                data_dict[key].append(np.nan)
                    df_curves = pd.DataFrame(data_dict)
                    print("df_curves",df_curves)

                    param_cols = df_meta.columns[3:]

                    df_params = df_meta[param_cols]

                    match_idx = None
                    print("df_params: ",df_params)
                    print(f"[DEBUG] Best params: {df_params.iterrows()}")
                    for idx, row in df_params.iterrows():
                        print(f"[DEBUG] Checking row {idx} for match with best_params: {best_params}")
                        row_values = row.values.astype(float)
                        print(f"row_values: ", row_values)
                        # Round both to 8 decimal places for comparison
                        rounded_row = np.round(row_values, 8)
                        print(f"rounded_row: ", rounded_row)
                        rounded_best = np.round(best_params, 8)
                        print(f"rounded_best: ", rounded_best)
                        print(f"[COMPARING] Comparing: params_from_metadata={rounded_row}, params_best={rounded_best}")
                        if np.array_equal(rounded_row, rounded_best):
                            print(f"[DEBUG] Match found at index: {idx}")
                            print(f"[DEBUG] Row values: {rounded_row}")
                            print(f"[DEBUG] Best params: {rounded_best}")
                            match_idx = idx
                            break

                    if match_idx is not None:
                        print(f"[DEBUG] Match index: {match_idx}")
                        print(f"[DEBUG] match_idx: {match_idx}")
                        print(f"[DEBUG] Available pressure columns: {df_curves.columns.tolist()}")
                        print(f"[DEBUG] Total number of pressure columns: {len([col for col in df_curves.columns if col.startswith('P_')])}")
                        pressure_cols = [col for col in df_curves.columns if col.startswith("P_")]
                        if match_idx < len(pressure_cols):
                            print(f"[DEBUG] Trying to access pressure column at index {match_idx}")
                            print(f"[DEBUG] pressure_cols[{match_idx}]: {pressure_cols[match_idx] if match_idx < len(pressure_cols) else 'IndexError'}")
                            col_name = pressure_cols[match_idx]
                            curve = df_curves[col_name].dropna().values
                            burntime = f"{len(curve) * time_step:.2f} s"
            except Exception as e:
                print(f"[WARN] Burntime calculation failed: {e}")
            # --- END NEW BURN TIME LOGIC BLOCK ---
            # param_lines.append(f"{'Burntime':<20}: {burntime}")
            param_lines.append(f"{'Output Dir':<20}: {output_dir}")
            if param_lines:
                param_text = "<b>Best Parameters:</b><br><pre>" + "\n".join(param_lines) + "</pre>"
            else:
                param_text = "<b>No best parameters available.</b>"
            print(f"[DEBUG] Setting param table text")
            print(f"[DEBUG] param_text={param_text}")
            print(f"[DEBUG] result keys: {result.keys()}")
            print(f"[DEBUG] best_x: {result.get('best_x')}")
            print(f"[DEBUG] optimization_variables: {result.get('optimization_variables')}")
            # Directly update param_table as in main tab
            print(f"[DEBUG] Calling setText directly, visible={self.param_table.isVisible()}, text={param_text}")
            self.param_table.setText(param_text)
            self.param_table.setVisible(True)
            self.param_table.adjustSize()

            # Optimization summary label
            summary_text = "<b>Optimization Summary:</b><br>"
            summary_text += f"<b>Burntime:</b> {burntime}<br>"
            summary_text += f"<b>Output Directory:</b> {output_dir}<br>"
            summary_label = QLabel()
            summary_label.setText(summary_text)
            summary_label.setStyleSheet("font-family: Courier; font-size: 13pt; padding: 6px;")
            self.results_layout.addWidget(summary_label)
            summary_label.adjustSize()

            self.results_tab.update()
            self.results_tab.repaint()
            QApplication.processEvents()
            print(f"[DEBUG] GOING INTO PLOT GEOMETRY WITH class_id={class_id}")
            self.plot_geometry(class_id, self.best_params)
            print(f"[DEBUG] !Skipped! Geometry plot done for class_id={class_id}")
            # Show zoomed plot if exists
            if self.zoomed_label:
                self.results_layout.removeWidget(self.zoomed_label)
                self.zoomed_label.setParent(None)
                self.zoomed_label = None
            plot_path = os.path.join(output_dir, "plot1.png")
            if os.path.exists(plot_path):
                from PyQt5.QtGui import QPixmap
                pixmap = QPixmap(plot_path)
                pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
                self.zoomed_label = QLabel()
                self.zoomed_label.setPixmap(pixmap)
                self.results_layout.insertWidget(0, self.zoomed_label)
            print(f"[DEBUG] Saving results to CSV")
            # self.save_results_csv(sim_time, sim_curve)
            print(f"[DEBUG] Skipped saving to CSV")
            # Switch to Results tab
            # QTimer.singleShot(0, lambda: self.tabs.setCurrentIndex(1))
            self.results_tab.update()
            self.results_tab.repaint()
            QApplication.processEvents()
        except Exception as e:
            self.status_label.setText("Optimization failed.")
            QMessageBox.critical(self, "Optimization Error", str(e))

    def cancel_optimization(self):
        self.cancel_flag.set()
        self.status_label.setText("Cancelling...")

    def poll_progress_queue(self):
        if not hasattr(self, "progress_queue") or self.progress_queue is None:
            return
        try:
            while not self.progress_queue.empty():
                msg = self.progress_queue.get_nowait()
                if isinstance(msg, dict):
                    self.progress_bar.setValue(int(msg.get("progress", 0)))
                    self.eta_label.setText(msg.get("eta", ""))
                elif isinstance(msg, tuple) and len(msg) == 2:
                    progress, total = msg
                    self.update_progress_bar(progress, total)
        except Exception as e:
            print(f"[WARN] Error polling progress queue: {e}")

def main():
    app = QApplication(sys.argv)
    gui = SRMGui()
    import helper_functions_GUI
    helper_functions_GUI.gui_instance = gui
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    def populate_results_summary(self, output_dir, result):
        try:
            metadata_path = result.get("metadata_csv_path")
            info_path = os.path.join(output_dir, "info.txt")
            burntime = "Unknown"
            iteration_count = 0
            time_step = 1.0
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                if "Iteration" in df.columns:
                    iteration_count = df["Iteration"].max()
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    for line in f:
                        if "time_step" in line:
                            try:
                                time_step = float(line.split(":")[-1].strip())
                            except:
                                pass
            if iteration_count and time_step:
                burntime = f"{iteration_count * time_step:.2f} s"
            summary_text = "<b>Optimization Summary:</b><br>"
            summary_text += f"<b>Burntime:</b> {burntime}<br>"
            summary_text += f"<b>Output Directory:</b> {output_dir}<br>"
            summary_label = QLabel()
            summary_label.setText(summary_text)
            summary_label.setStyleSheet("font-family: Courier; font-size: 13pt; padding: 6px;")
            self.results_layout.addWidget(summary_label)
        except Exception as e:
            print(f"[WARN] Could not populate results summary: {e}")
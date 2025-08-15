# Master Thesis: Solid Rocket Motor Optimization

This project implements a complete framework for optimizing the design of a solid rocket motor using OpenMotor simulations and a multi-variable optimization approach.

This repository also includes a classification pipeline to infer the grain geometry type (Bates, Star, or Endburner) from a pressure curve using a trained convolutional neural network.

## Features

- **Multi-variable Optimization**: Supports continuous and discrete variables.
- **Automated Simulation**: Uses OpenMotor CLI to run simulations for each candidate design.
- **Custom Constraints**: Includes penalties for design violations (e.g., throat > diameter).
- **Flexible Configuration**: Easily add/remove variables by editing a single list.
- **Postprocessing Tools**: Includes tools for visualizing pressure curve fits and comparing target/simulated outputs.
- **HDF5 Logging**: Archives optimization runs with pressure curves, parameter values, and metadata.
- **Classification Model**: Predicts grain geometry type (bates, star, or endburner) using a 1D CNN trained on pressure curves.
- **Integrated Pipeline**: Automatically classifies a pressure curve and runs the appropriate optimization script.

---

## Getting Started

### Requirements

- Python 3.10
- Dependencies listed in `requirements.txt`
- OpenMotor CLI available and accessible in system path
- Conda environment is recommended for easy installation of pygmo lib
- pytorch version 2.3.1

### Folder Structure

```
master_thesis/
├── openmotor/               # OpenMotor simulation engine
├── my_functions/            # Custom helper functions and optimizers
├── my_tool/                 # Postprocessing utilities
├── data/                    # Stores optimization results
├── motor.ric                # Template motor input file
├── main.py                  # Main entry point to run an optimization
├── save_data.py             # Script to store optimization runs in HDF5
├── postprocess_P_curve_comparison.py # Visualize target vs simulated curves
```

---

## Usage

### 0. Run Full Classification + Optimization Pipeline

You can use the integrated `run_pipeline.py` script to classify a pressure curve and automatically run the corresponding optimization.

#### How it works:
1. The script uses a 1D U-Net model to classify the geometry type of the motor from the input pressure curve.
2. Based on the predicted class:
   - Class 0 → Bates → Runs `the_main_bates.py`
   - Class 1 → Star → Runs `the_main_star.py`
   - Class 2 → Endburner → Runs `the_main_endburner.py`
3. It launches the corresponding optimization script as a subprocess.

#### Usage:

```bash
python my_tool/run_pipeline.py path/to/your/pressure_curve.csv
```

This will:
- Print the predicted class.
- Run the correct optimization script.
- Save results, plots, and metadata in the corresponding output directory.

#### Manual prediction usage:

If you want to just classify a pressure curve without running optimization:

```python
from master_thesis.my_functions.CNN.scripts.predictor import predict_class_from_file

predicted_class = predict_class_from_file("path/to/your/pressure_curve.csv")
print(predicted_class)  # Outputs 0, 1, or 2
```

> Make sure your pressure curve CSV contains a single row with pressure values (comma-separated).

### 5. Run with Graphical Interface

You can also use a graphical interface to classify a pressure curve and run the corresponding optimization with visual feedback.

#### Launch the GUI:

```bash
python my_tool/GUI_test/gui_launcher.py
```

#### Features:
- Choose a pressure curve CSV or enter a thrust value and burn time to generate one.
- Select population size and number of generations.
- View live updates of optimization progress (simulated curve vs. target).
- See predicted grain geometry.
- Save the pressure curve plot and geometry image.
- Export the final simulated pressure curve as a CSV.

#### How to Use:
1. **Input**:
   - Use the "Browse" button to select a pressure curve `.csv`, or
   - Enter a constant thrust value and burn time to auto-generate one.

2. **Settings**:
   - Enter the desired population size and number of generations for the optimizer.

3. **Run**:
   - Click **Run** to:
     - Classify the grain geometry (Bates, Star, or Endburner).
     - Start optimization using the corresponding script.
     - See a live plot of progress and final grain geometry.

4. **Export**:
   - Use **Save Curve Plot** and **Save Geometry Plot** to export PNG images.
   - Use **Save CSV** to export final simulation results.

> The GUI is ideal for demonstrations or manual inspection of different pressure curve targets.

### 1. Define Your Variables

In `the_main.py`:

```python
optimization_variables = [
    {
        "name": "diameter",
        "bounds": (0.02, 0.07),
        "label": "Outer Diameter (m)",
        "color": "teal",
        "type": "continuous"
    },
    {
        "name": "numPoints",
        "bounds": (4, 8),
        "label": "Number of Points",
        "color": "blue",
        "type": "discrete"
    },
    ...
]
```

### 2. Run the Optimization

```bash
python openmotor/the_main.py
```

### 3. Save Run to HDF5

```bash
python save_data.py
```

### 4. Compare Simulation vs Target

```bash
python postprocess_P_curve_comparison.py
```

---

## Customization

### Add a New Variable

Just add it to the `optimization_variables` list in `the_main.py`, and the rest of the system (plotting, optimization, HDF5 saving) will adapt automatically.

### Penalty Functions

Custom penalty logic (e.g., for geometry constraints or burn time mismatch) is located in `optimizer.py`. You can add or modify penalty terms here.

---

## Notes

- All generated plots and CSVs are stored under the `/data/optXX/` structure.
- Ensure `motor.ric` contains all variables you're optimizing for. A check is also included in `the\_main`.

---

## Author

Olivier Gianoli

---

## License

This project is part of a Master's Thesis and intended for academic use. Please cite appropriately if used.


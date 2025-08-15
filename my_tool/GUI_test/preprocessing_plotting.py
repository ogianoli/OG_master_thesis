import numpy as np
import matplotlib.pyplot as plt
from master_thesis.my_functions.CNN.config import INPUT_LENGTH, RESAMPLING

# RESAMPLING = True 
# INPUT_LENGTH = 1000

def normalize(arr):
    """Min-max normalize to range [0, 1]."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

def pad_or_crop(arr, length=INPUT_LENGTH):
    """Pad with zeros or crop to fixed length."""
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'constant')
    return arr[:length]

# --- Add resampling/interpolation function ---
def resample_to_length(arr, length=INPUT_LENGTH):
    """Resample the array to the target length using linear interpolation."""
    if len(arr) == length:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, length)
    return np.interp(x_new, x_old, arr)

def read_curves_from_file(file_path):
    """Read and preprocess each curve from a CSV file (one curve per row with a class ID at the end)."""
    # Import config for INPUT_LENGTH and RESAMPLING
    
    class_curves = {0: [], 1: [], 2: []}
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            if not row or len(row) < 2:
                continue  # Skip empty or invalid rows
            try:
                *values, class_id = [float(v) for v in row]
                class_id = int(class_id)
                if class_id not in class_curves:
                    continue
                values = np.array(values, dtype=np.float32)
                values = normalize(values)
                if RESAMPLING:
                    values = resample_to_length(values, length=INPUT_LENGTH)
                else:
                    values = pad_or_crop(values, length=INPUT_LENGTH)
                class_curves[class_id].append(values)
            except ValueError:
                continue  # Skip rows with non-numeric entries
    return class_curves

def plot_preprocessed_curves(class_curves):
    """Plot all preprocessed curves in separate subplots per class."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    class_titles = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
    for class_id, curves in class_curves.items():
        ax = axes[class_id]
        for curve in curves:
            ax.plot(curve, alpha=0.7)
        ax.set_title(f"Preprocessed Curves - {class_titles[class_id]}")
        ax.set_ylabel("Normalized Pressure")
        ax.grid(True)
    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "master_thesis/my_functions/CNN/data/functions_validation.csv"  # Replace with your real path
    class_curves = read_curves_from_file(file_path)
    plot_preprocessed_curves(class_curves)
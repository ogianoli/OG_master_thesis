import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.utils.dataset import FunctionDatasetWithResampling
from master_thesis.my_functions.CNN.config import VALIDATION_CSV_PATH, NUM_CLASSES, INPUT_LENGTH, RESAMPLING
import matplotlib.pyplot as plt
from master_thesis.my_functions.helper_functions import plot_geometry_and_curve_from_misclassified

model_num = 10
# RESAMPLING = True   # To override the config value

functions_csv_path_bates = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/pressure_curves.csv"  # update as needed
params_csv_path_bates = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/params.csv"        # update as needed
functions_csv_path_star = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/star/DOE_outputs/pressure_curves.csv"  # update as needed
params_csv_path_star = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/star/DOE_outputs/params.csv"        # update as needed
functions_csv_path_endburner = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/endburner/DOE_outputs/pressure_curves.csv"  # update as needed
params_csv_path_endburner = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/endburner/DOE_outputs/params.csv"        # update as needed

functions_csv_paths = [functions_csv_path_bates, functions_csv_path_star, functions_csv_path_endburner]
params_csv_paths = [params_csv_path_bates, params_csv_path_star, params_csv_path_endburner]

validation_csv_path = VALIDATION_CSV_PATH
class ValidationDataset(Dataset):
    def __init__(self, csv_path):
        self.X = []
        self.y = []

        with open(csv_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if len(row) < 2:
                    continue
                try:
                    *values, label = row
                    values = np.array([float(v) for v in values], dtype=np.float32)
                    label = int(label)

                    values = self.normalize(values)
                    values = self.pad_or_crop(values)

                    self.X.append(values)
                    self.y.append(label)
                except ValueError:
                    continue

        self.X = torch.tensor(self.X).unsqueeze(1)
        self.y = torch.tensor(self.y)

    def normalize(self, arr):
        y = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        return y

    def pad_or_crop(self, arr):
        if len(arr) < INPUT_LENGTH:
            pad = INPUT_LENGTH - len(arr)
            arr = np.pad(arr, (0, pad), 'constant')
        elif len(arr) > INPUT_LENGTH:
            arr = arr[:INPUT_LENGTH]
        return arr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_model_on_validation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    
    model = UNet1D(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(f"master_thesis/my_functions/CNN/models/unet1d_model_{model_num}.pth", map_location=device))
    model.eval()

    if RESAMPLING:
        dataset = FunctionDatasetWithResampling(VALIDATION_CSV_PATH)
    else: dataset = ValidationDataset(VALIDATION_CSV_PATH)
    loader = DataLoader(dataset, batch_size=1)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            # softmax(output)
            pred = torch.argmax(output, dim=1).item()
            all_preds.append(pred)
            all_labels.append(y.item())

    # Print misclassified examples
    print("\n❌ Misclassified Samples:")
    misclassified_indices = []
    misclassified_curves = []
    # Read validation CSV lines for true values
    with open(validation_csv_path, 'r') as f:
        val_lines = [line.strip() for line in f if line.strip()]
    for idx, (pred, true_label) in enumerate(zip(all_preds, all_labels)):
        if pred != true_label:
            # Get first 5 values from validation CSV, using idx as validation_index
            if idx >= len(val_lines):
                print(f"❌ Validation index {idx} out of range.")
                continue
            val_row = val_lines[idx].split(',')
            first_vals = tuple(np.array(val_row[:5], dtype=float).astype(str))
            print(f"Index {idx} - Pred: {pred}, True: {true_label}, First 5 values: {first_vals}")
            misclassified_indices.append(idx)
            misclassified_curves.append(dataset.X[idx][0].numpy())

    if misclassified_indices:
        print("\n📉 Plotting all misclassified curves in one figure...")
        plt.figure(figsize=(10, 4))
        for idx, (pred, true_label) in enumerate(zip(all_preds, all_labels)):
            if pred != true_label:
                curve = dataset.X[idx][0].numpy()
                plt.plot(curve, label=f'Idx {idx} (Pred: {pred}, True: {true_label})')
        plt.title("Misclassified Curves")
        plt.xlabel("Time index")
        plt.ylabel("Normalized Pressure")
        plt.legend(fontsize='small', loc='upper right', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot geometry and curve for each misclassified sample ---
        from master_thesis.my_functions.helper_functions import reverse_preprocessing
        for i, curve in enumerate(misclassified_curves):
            original_like_curve = reverse_preprocessing(curve)
            plot_geometry_and_curve_from_misclassified(functions_csv_paths, params_csv_paths, idx=i, validation_csv=validation_csv_path, validation_index=misclassified_indices[i])
    else:
        print("\n✅ No misclassified curves found.")

    if all_labels and all_preds:
        acc = accuracy_score(all_labels, all_preds)
        correct_count = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        print(f"\n✅ Validation Accuracy: {acc:.4f} ({correct_count}/{len(all_labels)} correct)")
    else:
        print("\n⚠️ Accuracy calculation skipped: no predictions were made.")

    if all_labels and all_preds:
        report = classification_report(all_labels, all_preds, digits=4)
        print("\n📊 Classification Report:")
        print(report)
    else:
        print("\n⚠️ Classification report skipped: no predictions were made.")


if __name__ == "__main__":
    evaluate_model_on_validation()
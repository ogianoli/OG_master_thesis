import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.utils.dataset import FunctionDatasetWithResampling
from master_thesis.my_functions.CNN.config import VALIDATION_CSV_PATH, NUM_CLASSES, INPUT_LENGTH, RESAMPLING
import pandas as pd

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

def run_prediction_campaign():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = []

    for model_num in range(1, 11):
        model = UNet1D(n_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(f"master_thesis/my_functions/CNN/models/unet1d_model_{model_num}.pth", map_location=device))
        model.eval()

        if RESAMPLING:
            dataset = FunctionDatasetWithResampling(VALIDATION_CSV_PATH)
        else:
            dataset = ValidationDataset(VALIDATION_CSV_PATH)

        loader = DataLoader(dataset, batch_size=1)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
                all_preds.append(pred)
                all_labels.append(y.item())

        if all_labels and all_preds:
            acc = accuracy_score(all_labels, all_preds)
            correct_count = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
            total = len(all_labels)
            results.append({
                "Model": model_num,
                "Accuracy": round(acc, 4),
                "Correct": correct_count,
                "Total": total
            })

    df = pd.DataFrame(results)
    print("\n📊 Summary of Model Performance:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_prediction_campaign()

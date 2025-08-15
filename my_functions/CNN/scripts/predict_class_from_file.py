import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.config import NUM_CLASSES, INPUT_LENGTH

model_num = 10  # Set the model number you'd like to use

class InferenceDataset(Dataset):
    def __init__(self, csv_path):
        self.X = []

        with open(csv_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if len(row) < 1:
                    continue
                try:
                    values = np.array([float(v) for v in row], dtype=np.float32)
                    values = self.normalize(values)
                    values = self.pad_or_crop(values)
                    self.X.append(values)
                except ValueError:
                    continue

        self.X = torch.tensor(self.X).unsqueeze(1)

    def normalize(self, arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

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
        return self.X[idx]

def predict_class_from_file(csv_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet1D(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        f"master_thesis/my_functions/CNN/models/unet1d_model_{model_num}.pth", 
        map_location=device
    ))
    model.eval()

    dataset = InferenceDataset(csv_path)
    loader = DataLoader(dataset, batch_size=1)

    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
            preds.append(pred)

    if preds:
        return max(set(preds), key=preds.count)
    return None
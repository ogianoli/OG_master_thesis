import torch
from torch.utils.data import Dataset
import numpy as np
from master_thesis.my_functions.CNN.config import INPUT_LENGTH

class FunctionDataset(Dataset):
    def __init__(self, csv_path):
        self.X = []
        self.y = []

        with open(csv_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if len(row) < 2:
                    continue  # skip empty or invalid rows

                try:
                    *values, label = row
                    values = np.array([float(v) for v in values], dtype=np.float32)
                    label = int(label)

                    values = self.normalize(values)
                    values = self.pad_or_crop(values)

                    self.X.append(values)
                    self.y.append(label)
                except ValueError:
                    continue  # skip rows with non-numeric data

        self.X = torch.tensor(self.X).unsqueeze(1)  # Shape: (N, 1, L)
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
    
class FunctionDatasetForPrediction(Dataset):
    def __init__(self, csv_path):
        self.X = []

        with open(csv_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if not row:
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
        y = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        return y

    def pad_or_crop(self, arr):
        if len(arr) < INPUT_LENGTH:
            return np.pad(arr, (0, INPUT_LENGTH - len(arr)), 'constant')
        return arr[:INPUT_LENGTH]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
class FunctionDatasetWithResampling(Dataset):
    def __init__(self, csv_path):
        self.X = []
        self.y = []

        with open(csv_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                if len(row) < 2:
                    continue  # skip empty or invalid rows

                try:
                    *values, label = row
                    values = np.array([float(v) for v in values], dtype=np.float32)
                    label = int(label)

                    values = self.normalize_and_resample(values)

                    self.X.append(values)
                    self.y.append(label)
                except ValueError:
                    continue  # skip rows with non-numeric data

        self.X = torch.tensor(self.X).unsqueeze(1)  # Shape: (N, 1, L)
        self.y = torch.tensor(self.y)

    def normalize_and_resample(self, arr):
        # Normalize Y-axis
        y_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        # Resample to fixed INPUT_LENGTH using interpolation
        if len(y_norm) == 1:
            return np.repeat(y_norm, INPUT_LENGTH)
        old_x = np.linspace(0, 1, num=len(y_norm))
        new_x = np.linspace(0, 1, num=INPUT_LENGTH)
        return np.interp(new_x, old_x, y_norm).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class InferenceDataset(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        self.X = []

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Chamber Pressure(Pa)" not in df.columns:
            raise ValueError("CSV must contain a 'Chamber Pressure(Pa)' column.")

        values = df["Chamber Pressure(Pa)"].values.astype(np.float32)
        values = self.normalize(values)
        values = self.pad_or_crop(values)
        self.X.append(values)

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


class InferenceDatasetWithResampling(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        self.X = []

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Chamber Pressure(Pa)" not in df.columns:
            raise ValueError("CSV must contain a 'Chamber Pressure(Pa)' column.")

        values = df["Chamber Pressure(Pa)"].values.astype(np.float32)
        values = self.normalize_and_resample(values)
        self.X.append(values)

        self.X = torch.tensor(self.X).unsqueeze(1)

    def normalize_and_resample(self, arr):
        y_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        if len(y_norm) == 1:
            return np.repeat(y_norm, INPUT_LENGTH)
        old_x = np.linspace(0, 1, num=len(y_norm))
        new_x = np.linspace(0, 1, num=INPUT_LENGTH)
        return np.interp(new_x, old_x, y_norm).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
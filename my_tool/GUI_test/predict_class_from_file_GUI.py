import torch
from torch.utils.data import DataLoader
import numpy as np
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.config import NUM_CLASSES, INPUT_LENGTH
from master_thesis.my_functions.CNN.config import RESAMPLING
from master_thesis.my_functions.CNN.utils.dataset import InferenceDataset, InferenceDatasetWithResampling

model_num = 9  # Set the model number you'd like to use

def predict_class_from_file(csv_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet1D(n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        f"master_thesis/my_functions/CNN/models/unet1d_model_{model_num}.pth", 
        map_location=device
    ))
    model.eval()

    dataset_class = InferenceDatasetWithResampling if RESAMPLING else InferenceDataset
    dataset = dataset_class(csv_path)
    loader = DataLoader(dataset, batch_size=1)

    preds = []
    probs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            output = model(x)
            prob_values = torch.softmax(output, dim=1).cpu().numpy()[0]
            top1 = int(np.argmax(prob_values))
            preds.append(top1)
            probs = [(i, float(p)) for i, p in enumerate(prob_values)]
            probs.sort(key=lambda x: x[1], reverse=True)

    print(f"Predicted class: {preds[0]} with probabilities: {probs}")

    if preds:
        return preds[0], probs
    return None, []
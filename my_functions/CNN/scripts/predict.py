import torch
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.utils.dataset import FunctionDatasetForPrediction
from master_thesis.my_functions.CNN.config import *

def predict():
    model = UNet1D(n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
    model.eval()

    dataset = FunctionDatasetForPrediction(VALIDATION_CSV_PATH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print(f"Loaded dataset from {VALIDATION_CSV_PATH} with {len(dataset)} samples")

    with torch.no_grad():
        print("Starting predictions...")
         # Iterate through the dataset and make predictions
         # Assuming the dataset returns (input, label) tuples
         # Here we only care about the input for prediction
         # and we will print the predicted class for each sample
        print("loader content: ", loader.dataset)
        print("Number of samples in loader: ", len(loader))
        with torch.no_grad():
            for i, x in enumerate(loader):
                out = model(x)
                pred = torch.argmax(out, dim=1).item()
                probs = torch.softmax(out, dim=1)
                prob = round(probs[0, pred].item(), 3)  # get probability of predicted class
                other_probs = {cls: round(probs[0, cls].item(), 3) for cls in range(probs.shape[1]) if cls != pred}
                print(f"Row {i}: Predicted class {pred} with probability {prob}, other classes probabilities: {other_probs}")

if __name__ == "__main__":
    predict()
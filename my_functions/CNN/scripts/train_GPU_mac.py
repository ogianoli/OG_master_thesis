import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from master_thesis.my_functions.CNN.models.unet_1d import UNet1D
from master_thesis.my_functions.CNN.utils.dataset import FunctionDataset, FunctionDatasetWithResampling
from master_thesis.my_functions.CNN.config import *
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

def train():

    train_num = 10
    RESAMPLING = True  # Overwriting the config so i can run it in parallel with different settings
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet1D(n_classes=NUM_CLASSES).to(device)

    dataset_class = FunctionDatasetWithResampling if RESAMPLING else FunctionDataset
    dataset = dataset_class(TRAIN_TEST_CSV_PATH)

    # --- Split into 80% train, 20% test ---
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- TensorBoard for logging ---
    writer = SummaryWriter(log_dir=f"master_thesis/my_functions/CNN/tensorboard/runs/unet1d_training_{train_num}")

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Evaluate on test set and log to TensorBoard ---
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_test_loss = test_loss / len(test_loader)

        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Epoch {epoch+1}, Test Accuracy: {acc:.4f}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"master_thesis/my_functions/CNN/models/unet1d_model_{train_num}.pth")
            print(f"✅ Saved new best model with test loss: {best_loss:.4f} at epoch {best_epoch}")

    writer.close()

if __name__ == "__main__":
    train()
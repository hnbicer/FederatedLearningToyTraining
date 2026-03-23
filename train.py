from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_quadrant_dataset import MNISTQuadrantDataset
from model import QuadrantVFLModel


# ============================================================
# Config
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "data"

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
LATENT_DIM = 8
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for q1, q2, q3, q4, y in loader:
        q1 = q1.to(device)
        q2 = q2.to(device)
        q3 = q3.to(device)
        q4 = q4.to(device)
        y = y.to(device)

        logits = model(q1, q2, q3, q4)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc


# ============================================================
# Main
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    train_dataset = MNISTQuadrantDataset(
        root=DATA_DIR,
        train=True,
        download=True,
    )
    test_dataset = MNISTQuadrantDataset(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = QuadrantVFLModel(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_train_samples = 0

        for q1, q2, q3, q4, y in train_loader:
            q1 = q1.to(DEVICE)
            q2 = q2.to(DEVICE)
            q3 = q3.to(DEVICE)
            q4 = q4.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(q1, q2, q3, q4)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total_train_samples += y.size(0)

        train_loss = running_loss / total_train_samples
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {100.0 * test_acc:.2f}%"
        )

    save_path = Path(__file__).resolve().parent / "quadrant_vfl_mnist.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
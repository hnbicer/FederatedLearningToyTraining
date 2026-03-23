from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_quadrant_dataset import MNISTQuadrantDataset
from model import QuadrantVFLModel


# ============================================================
# Config
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
LATENT_DIMS = [2, 4, 8, 16, 32, 64]
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# One training run for one latent dimension
# ============================================================
def train_one_model(latent_dim, train_loader, test_loader, device):
    model = QuadrantVFLModel(
        latent_dim=latent_dim,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_train_samples = 0

        for q1, q2, q3, q4, y in train_loader:
            q1 = q1.to(device)
            q2 = q2.to(device)
            q3 = q3.to(device)
            q4 = q4.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(q1, q2, q3, q4)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total_train_samples += y.size(0)

        epoch_train_loss = running_loss / total_train_samples
        _, epoch_test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"[latent_dim={latent_dim:>2}] "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Test Acc: {100.0 * epoch_test_acc:.2f}%"
        )

    # Final full evaluation
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return model, train_loss, train_acc, test_loss, test_acc


# ============================================================
# Main
# ============================================================
def main():
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

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

    rows_acc = []
    rows_full = []

    for latent_dim in LATENT_DIMS:
        print("\n" + "=" * 70)
        print(f"Training model with latent_dim = {latent_dim}")
        print("=" * 70)

        model, train_loss, train_acc, test_loss, test_acc = train_one_model(
            latent_dim=latent_dim,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
        )

        # Save model for this latent size
        model_path = OUT_DIR / f"quadrant_vfl_mnist_latent{latent_dim}.pt"
        torch.save(model.state_dict(), model_path)

        print(
            f"Final Results | latent_dim={latent_dim:>2} | "
            f"Train Acc: {100.0 * train_acc:.2f}% | "
            f"Test Acc: {100.0 * test_acc:.2f}%"
        )

        # Requested compact array
        rows_acc.append([latent_dim, train_acc, test_acc])

        # More detailed array
        rows_full.append([latent_dim, train_loss, train_acc, test_loss, test_acc])

    results_acc = np.array(rows_acc, dtype=np.float32)
    results_full = np.array(rows_full, dtype=np.float32)

    # Save as .npy
    acc_npy_path = OUT_DIR / "latent_sweep_acc.npy"
    full_npy_path = OUT_DIR / "latent_sweep_full.npy"

    np.save(acc_npy_path, results_acc)
    np.save(full_npy_path, results_full)

    # Save as .csv for easy reading
    acc_csv_path = OUT_DIR / "latent_sweep_acc.csv"
    full_csv_path = OUT_DIR / "latent_sweep_full.csv"

    np.savetxt(
        acc_csv_path,
        results_acc,
        delimiter=",",
        header="latent_dim,train_acc,test_acc",
        comments="",
    )

    np.savetxt(
        full_csv_path,
        results_full,
        delimiter=",",
        header="latent_dim,train_loss,train_acc,test_loss,test_acc",
        comments="",
    )

    print("\n" + "=" * 70)
    print("Compact results array: [latent_dim, train_acc, test_acc]")
    print(results_acc)

    print("\nDetailed results array: [latent_dim, train_loss, train_acc, test_loss, test_acc]")
    print(results_full)

    print(f"\nSaved:")
    print(f"  {acc_npy_path}")
    print(f"  {full_npy_path}")
    print(f"  {acc_csv_path}")
    print(f"  {full_csv_path}")


if __name__ == "__main__":
    main()
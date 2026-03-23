from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).resolve().parent
    results_path = base_dir / "latent_sweep_acc.npy"

    results = np.load(results_path)

    # results columns:
    # [latent_dim, train_acc, test_acc]
    latent_dims = results[:, 0]
    train_accs = results[:, 1]
    test_accs = results[:, 2]

    plt.figure(figsize=(7, 5))
    plt.plot(latent_dims, test_accs, marker="o", label="Test Accuracy")
    plt.xlabel("Latent Dimension per Sensor")
    plt.ylabel("Accuracy")
    plt.title("Latent Dimension vs Test Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = base_dir / "latent_dim_vs_test_acc.png"
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
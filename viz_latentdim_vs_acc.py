from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Set to None to plot all latent dims
PLOT_DIMS = [1,2,4,16,32]


def main():
    base_dir = Path(__file__).resolve().parent
    results_path = base_dir / "latent_sweep_acc.npy"

    results = np.load(results_path)

    # expected columns:
    # [latent_dim, train_acc, final_test_acc, best_test_acc]
    latent_dims = results[:, 0]
    best_test_accs = results[:, 3]

    if PLOT_DIMS is not None:
        plot_dims_set = set(PLOT_DIMS)
        mask = np.array([dim in plot_dims_set for dim in latent_dims])
        latent_dims = latent_dims[mask]
        best_test_accs = best_test_accs[mask]

        if len(latent_dims) == 0:
            raise ValueError(
                f"No matching latent dimensions found in results for PLOT_DIMS={PLOT_DIMS}"
            )

    # sort by latent dimension
    sort_idx = np.argsort(latent_dims)
    latent_dims = latent_dims[sort_idx]
    best_test_accs = best_test_accs[sort_idx]

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()

    ax.plot(
        latent_dims,
        best_test_accs,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Best Test Accuracy",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(latent_dims)
    ax.set_xticklabels([str(int(x)) for x in latent_dims], fontsize=14)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(axis="y", labelsize=14)

    y_min = max(0.0, best_test_accs.min() - 0.003)
    y_max = min(1.0, best_test_accs.max() + 0.003)
    ax.set_ylim(y_min, y_max)

    for x, y in zip(latent_dims, best_test_accs):
        ax.annotate(
            f"{100*y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=13,
        )

    ax.set_xlabel("Latent Dimension per Sensor", fontsize=16)
    ax.set_ylabel("Best Test Accuracy", fontsize=16)
    ax.set_title("Latent Dimension vs Best Test Accuracy", fontsize=18)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    plt.tight_layout()

    if PLOT_DIMS is None:
        out_path = base_dir / "latent_dim_vs_best_test_acc_pretty.png"
    else:
        dims_str = "_".join(str(int(d)) for d in latent_dims)
        out_path = base_dir / f"latent_dim_vs_best_test_acc_pretty_{dims_str}.png"

    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
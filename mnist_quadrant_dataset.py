from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTQuadrantDataset(Dataset):
    """
    Returns:
        q1, q2, q3, q4, label

    Each qk is a tensor of shape [1, 14, 14].
    MNIST original image shape is [1, 28, 28].
    """

    def __init__(self, root="data", train=True, download=False):
        self.transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(
            root=Path(root),
            train=train,
            download=download,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]  # x: [1, 28, 28]

        q1 = x[:, 0:14, 0:14]     # top-left
        q2 = x[:, 0:14, 14:28]    # top-right
        q3 = x[:, 14:28, 0:14]    # bottom-left
        q4 = x[:, 14:28, 14:28]   # bottom-right

        return q1, q2, q3, q4, y


if __name__ == "__main__":
    ds = MNISTQuadrantDataset(root="data", train=True, download=True)
    q1, q2, q3, q4, y = ds[0]

    print("q1 shape:", q1.shape)
    print("q2 shape:", q2.shape)
    print("q3 shape:", q3.shape)
    print("q4 shape:", q4.shape)
    print("label:", y)
from pathlib import Path

from torchvision import datasets, transforms


def main():
    data_dir = Path(__file__).resolve().parent / "data"

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    print(f"MNIST downloaded to: {data_dir}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    x, y = train_dataset[0]
    print(f"One sample shape: {x.shape}")   # should be torch.Size([1, 28, 28])
    print(f"One sample label: {y}")


if __name__ == "__main__":
    main()
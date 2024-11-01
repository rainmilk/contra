import json
import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
from configs import settings
from gen_dataset.split_dataset import split_by_class, sample_class_balanced_data

conference_name = "cvpr"


def create_flower102_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.25,
    split_ratio=0.6,
):
    rng = np.random.default_rng(42)  # Using a new random number generator with a seed

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    # Load FLOWER-102 dataset
    train_dataset = datasets.Flowers102(
        root=data_dir, split="train", download=False, transform=data_transform
    )
    test_dataset = datasets.Flowers102(
        root=data_dir, split="test", download=False, transform=data_transform
    )

    print("Using class-balanced data splitting...")
    dataset_name = "flower-102"
    num_classes = 102

    # Convert train dataset to numpy format
    train_data = [train_dataset[i][0].numpy() for i in range(len(train_dataset))]
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Split training data by class
    class_data = split_by_class(train_data, train_labels, num_classes)

    # Create class-balanced D_0 and D_inc datasets
    D_0_data, D_0_labels, D_inc_data, D_inc_labels = sample_class_balanced_data(
        class_data, split_ratio
    )

    # D_1_plus: Adding noise
    num_noisy_samples = int(len(D_inc_labels) * noise_ratio)
    noisy_indices = rng.choice(len(D_inc_labels), num_noisy_samples, replace=False)
    noisy_sel = np.zeros(len(D_inc_labels), dtype=bool)
    noisy_sel[noisy_indices] = True

    D_noisy_data = D_inc_data[noisy_sel]
    D_noisy_true_labels = D_inc_labels[noisy_sel]

    D_normal_data = D_inc_data[~noisy_sel]
    D_normal_labels = D_inc_labels[~noisy_sel]

    if noise_type == "symmetric":
        D_noisy_labels = rng.choice(num_classes, num_noisy_samples, replace=True)
    else:
        raise ValueError("Invalid noise type.")

    # Save dataset
    save_path = os.path.join(
        gen_dir, f"nr_{noise_ratio}_nt_{noise_type}_{conference_name}"
    )
    os.makedirs(save_path, exist_ok=True)

    # Save datasets
    D_1_minus_data_path = os.path.join(save_path, "train_clean_data.npy")
    D_1_minus_labels_path = os.path.join(save_path, "train_clean_labels.npy")
    D_1_plus_data_path = os.path.join(save_path, "train_noisy_data.npy")
    D_1_plus_labels_path = os.path.join(save_path, "train_noisy_labels.npy")
    D_1_plus_true_labels_path = os.path.join(save_path, "train_noisy_true_labels.npy")

    np.save(D_1_minus_data_path, np.array(D_normal_data))
    np.save(D_1_minus_labels_path, np.array(D_normal_labels))
    np.save(D_1_plus_data_path, np.array(D_noisy_data))
    np.save(D_1_plus_labels_path, np.array(D_noisy_labels))
    np.save(D_1_plus_true_labels_path, np.array(D_noisy_true_labels))

    print("D_0, D_1_minus, and D_1_plus datasets have been generated and saved.")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate FLOWER-102 experimental dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(settings.root_dir, "data/flower-102/normal"),
        required=True,
        help="Directory of the original FLOWER-102 dataset",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        default=os.path.join(settings.root_dir, "data/flower-102/gen/"),
        help="Directory to save the generated dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["flower-102"],
        default="flower-102",
        help="Dataset (only 'flower-102' is supported)",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.6, help="Train split ratio (default 0.6)"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric"],
        default="symmetric",
        help="Label noise type (currently only 'symmetric' is supported)",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.25, help="Noise ratio (default 0.25)"
    )

    args = parser.parse_args()

    create_flower102_npy_files(
        args.data_dir,
        args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()

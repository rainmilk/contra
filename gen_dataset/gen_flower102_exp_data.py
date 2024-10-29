import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
import json

import collections
from configs import settings
from split_dataset import split


def create_flower102_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    split_ratio=0.5,
    retention_ratios=[0.5, 0.3, 0.1],
):
    rng = np.random.default_rng(42)  # 使用新的随机数生成器并设置种子

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整所有图像为 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    # 加载 FLOWER-102 数据集
    train_dataset = datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=data_transform
    )
    test_dataset = datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=data_transform
    )

    # 保存原始训练数据集
    train_data_path = os.path.join(data_dir, "train_data.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    train_data = [train_dataset[i][0].numpy() for i in range(len(train_dataset))]
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    np.save(train_data_path, np.array(train_data))
    np.save(train_labels_path, np.array(train_labels))

    # 划分训练集为 D_0 和 D_1
    num_train_samples = len(train_dataset)
    indices = np.arange(num_train_samples)
    rng.shuffle(indices)
    split_point = int(split_ratio * num_train_samples)
    D_0_indices = indices[:split_point]
    D_1_indices = indices[split_point:]

    # 保存 D_0 和 D_1 的索引
    D_0_indices_path = os.path.join(gen_dir, "D_0_indices.npy")
    D_1_indices_path = os.path.join(gen_dir, "D_1_indices.npy")
    np.save(D_0_indices_path, D_0_indices)
    np.save(D_1_indices_path, D_1_indices)

    assert (
        len(set(D_0_indices).intersection(set(D_1_indices))) == 0
    ), "D_0 和 D_1 的划分存在重叠"

    # 生成 D_1_minus 和 D_1_plus 数据集
    D_1_data = [train_data[i] for i in D_1_indices]
    D_1_labels = [train_labels[i] for i in D_1_indices]

    # 添加噪声到 D_1 数据集 (生成 D_1_plus)
    num_noisy_samples = int(len(D_1_labels) * noise_ratio)
    noisy_indices = rng.choice(len(D_1_labels), num_noisy_samples, replace=False)
    D_1_plus_labels = np.array(D_1_labels, copy=True)

    for idx in noisy_indices:
        original_label = D_1_plus_labels[idx]
        possible_labels = list(set(range(102)) - {original_label})
        D_1_plus_labels[idx] = rng.choice(possible_labels)

    # 保存 D_1_minus 和 D_1_plus
    D_1_minus_data_path = os.path.join(gen_dir, "D_1_minus_data.npy")
    D_1_minus_labels_path = os.path.join(gen_dir, "D_1_minus_labels.npy")
    D_1_plus_data_path = os.path.join(gen_dir, "D_1_plus_data.npy")
    D_1_plus_labels_path = os.path.join(gen_dir, "D_1_plus_labels.npy")

    np.save(D_1_minus_data_path, np.array(D_1_data))
    np.save(D_1_minus_labels_path, np.array(D_1_labels))
    np.save(D_1_plus_data_path, np.array(D_1_data))
    np.save(D_1_plus_labels_path, np.array(D_1_plus_labels))

    print("D_0、D_1_minus 和 D_1_plus 数据集已生成并保存。")


def main():
    parser = argparse.ArgumentParser(description="生成 FLOWER-102 实验数据集。")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="原始 FLOWER-102 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir", type=str, required=True, help="生成数据集的保存目录"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.5, help="D_0 和 D_1 的划分比例"
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="D_1 中噪声样本的比例"
    )

    args = parser.parse_args()

    create_flower102_npy_files(
        args.data_dir,
        args.gen_dir,
        split_ratio=args.split_ratio,
        noise_ratio=args.noise_ratio,
    )


if __name__ == "__main__":
    main()

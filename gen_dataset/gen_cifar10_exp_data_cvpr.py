import json
import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
from configs import settings

conference_name = "cvpr"


def create_dataset_files(
    data_dir,
    gen_dir,
    dataset_name="cifar-10",
    noise_type="symmetric",
    noise_ratio=0.5,
    split_ratio=0.5,
):
    rng = np.random.default_rng(42)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            (
                transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261])
                if dataset_name == "cifar-10"
                else transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ),
        ]
    )

    # 加载数据集
    if dataset_name == "cifar-10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == "cifar-100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        num_classes = 100
    elif dataset_name == "pets-37":
        train_dataset = datasets.OxfordIIITPet(
            root=data_dir, split="trainval", download=True, transform=transform
        )
        num_classes = 37
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    # 划分训练集为 D_0 和 D_1
    num_train_samples = len(train_dataset)
    indices = np.arange(num_train_samples)
    rng.shuffle(indices)
    split_point = int(split_ratio * num_train_samples)
    D_0_indices = indices[:split_point]
    D_1_indices = indices[split_point:]

    # D_0：无噪声部分
    D_0_data = [train_dataset[i][0].numpy() for i in D_0_indices]
    D_0_labels = [train_dataset[i][1] for i in D_0_indices]

    # D_1_minus：无噪声部分
    D_1_minus_data = [train_dataset[i][0].numpy() for i in D_1_indices]
    D_1_minus_labels = [train_dataset[i][1] for i in D_1_indices]

    # D_1_plus：添加噪声
    D_1_plus_data = D_1_minus_data.copy()
    D_1_plus_labels = D_1_minus_labels.copy()
    num_noisy_samples = int(len(D_1_minus_labels) * noise_ratio)
    noisy_indices = rng.choice(len(D_1_minus_labels), num_noisy_samples, replace=False)

    if noise_type == "symmetric":
        for idx in noisy_indices:
            original_label = D_1_plus_labels[idx]
            possible_labels = list(set(range(num_classes)) - {original_label})
            D_1_plus_labels[idx] = rng.choice(possible_labels)
    else:
        raise ValueError("Invalid noise type.")

    # 保存数据集
    save_path = os.path.join(
        gen_dir, f"nr_{noise_ratio}_nt_{noise_type}_{conference_name}"
    )
    os.makedirs(save_path, exist_ok=True)

    # 保存数据集
    D_0_data_path = os.path.join(save_path, "D_0_data.npy")
    D_0_labels_path = os.path.join(save_path, "D_0_labels.npy")
    np.save(D_0_data_path, np.array(D_0_data))
    np.save(D_0_labels_path, np.array(D_0_labels))

    D_1_minus_data_path = os.path.join(save_path, "D_1_minus_data.npy")
    D_1_minus_labels_path = os.path.join(save_path, "D_1_minus_labels.npy")
    np.save(D_1_minus_data_path, np.array(D_1_minus_data))
    np.save(D_1_minus_labels_path, np.array(D_1_minus_labels))

    D_1_plus_data_path = os.path.join(save_path, "D_1_plus_data.npy")
    D_1_plus_labels_path = os.path.join(save_path, "D_1_plus_labels.npy")
    np.save(D_1_plus_data_path, np.array(D_1_plus_data))
    np.save(D_1_plus_labels_path, np.array(D_1_plus_labels))

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_data = [test_dataset[i][0].numpy() for i in range(len(test_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

    # 保存测试数据集
    test_data_path = os.path.join(save_path, "test_data.npy")
    test_labels_path = os.path.join(save_path, "test_label.npy")
    np.save(test_data_path, np.array(test_data))
    np.save(test_labels_path, np.array(test_labels))


    print("D_0、D_1_minus 和 D_1_plus 数据集已生成并保存。")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-10/normal",
        help="原始 CIFAR-10 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-10/gen/",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["cifar-10", "cifar-100", "pets-37"],
        default="cifar-10",
        help="数据集名称：'cifar-10', 'cifar-100', 或 'pets-37'",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="标签噪声类型：'symmetric' 或 'asymmetric'",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.5, help="噪声比例（默认 0.5）"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.5, help="训练集划分比例（默认 0.5）"
    )

    args = parser.parse_args()

    create_dataset_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        dataset_name=args.dataset_name,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()

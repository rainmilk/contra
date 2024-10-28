import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
import json

import collections
from configs import settings
from split_dataset import split


def load_classes_from_file(file_path):
    """从文件中读取类别列表"""
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_flower102_superclass_mapping(file_path):
    """从JSON文件中加载 FLOWER-102 的 superclass 与 child class 的映射"""
    with open(file_path, "r") as f:
        flower102_superclass_to_child = json.load(f)
    return flower102_superclass_to_child


def create_flower102_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
    balanced=False,
    split_ratio=0.5,
):

    rng = np.random.default_rng(42)

    data_transform = transforms.Compose(
        [
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

    case = settings.get_case(noise_ratio, noise_type, balanced)

    print("使用类均衡的数据划分方式...")
    dataset_name = "flower-102"
    num_classes = 102
    D_inc_data, D_inc_labels = split(
        dataset_name, case, train_dataset, test_dataset, num_classes
    )

    # 读取 FLOWER-102 类别
    flower102_classes_file = os.path.join(
        settings.root_dir, "configs/classes/flower_102_classes.txt"
    )
    flower102_classes = load_classes_from_file(flower102_classes_file)

    # 读取 FLOWER-102 的 superclass 和 child class 映射
    flower102_mapping_file = os.path.join(
        settings.root_dir, "configs/classes/flower_102_mapping.json"
    )
    flower102_superclass_mapping = load_flower102_superclass_mapping(
        flower102_mapping_file
    )

    # 打印读取到的类别信息
    print("FLOWER-102 Classes:", flower102_classes)

    # 定义遗忘类别和噪声类别
    forget_classes = list(range(50))  # 前50个类别作为遗忘类别
    noise_classes = list(range(50, 75))  # 后面的25个类别作为噪声类别

    # 获取增量数据集中的遗忘类别和噪声类别的索引
    D_inc_forget_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in forget_classes
    ]
    D_inc_noise_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in noise_classes
    ]

    symmetric_noisy_classes = []
    symmetric_noisy_classes_simple = set()

    rng = np.random.default_rng(42)

    # 生成增量版本数据集
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # 模拟遗忘：根据保留比例抽取遗忘类别的样本
        num_forget_samples = int(len(D_inc_forget_indices) * retention_ratio)

        if num_forget_samples > 0:
            forget_sample_indices = rng.choice(
                D_inc_forget_indices, num_forget_samples, replace=False
            )
            D_f_data = D_inc_data[forget_sample_indices]
            D_f_labels = D_inc_labels[forget_sample_indices]
        else:
            D_f_data = torch.empty(0, 3, 224, 224)
            D_f_labels = torch.empty(0, dtype=torch.long)

        # 噪声注入：对噪声类别的样本注入噪声
        noise_sample_indices = D_inc_noise_indices
        num_noisy_samples = int(len(noise_sample_indices) * noise_ratio)

        if num_noisy_samples > 0:
            noisy_indices = rng.choice(
                noise_sample_indices, num_noisy_samples, replace=False
            )
        else:
            noisy_indices = []

        D_n_data = D_inc_data[noise_sample_indices]
        D_n_labels = D_inc_labels[noise_sample_indices]

        for idx_in_D_n, D_inc_idx in enumerate(noise_sample_indices):
            if D_inc_idx in noisy_indices:
                original_label = int(D_n_labels[idx_in_D_n].item())

                # 确保仅对 noise_classes 中的类别进行替换
                if original_label in noise_classes:
                    original_class_name = flower102_classes[original_label]
                    original_superclass = next(
                        (
                            superclass
                            for superclass, children in flower102_superclass_mapping.items()
                            if original_class_name in children
                        ),
                        None,
                    )

                if noise_type == "symmetric":
                    new_label = rng.choice(
                        [i for i in range(102) if i != original_label]
                    )
                    D_n_labels[idx_in_D_n] = new_label
                    new_class_name = flower102_classes[new_label]
                    new_superclass = next(
                        (
                            superclass
                            for superclass, children in flower102_superclass_mapping.items()
                            if new_class_name in children
                        ),
                        None,
                    )
                    symmetric_noisy_classes.append(
                        {
                            "original_label": int(original_label),
                            "original_class_name": original_class_name,
                            "original_superclass": original_superclass,
                            "new_label": int(new_label),
                            "new_class_name": new_class_name,
                            "new_superclass": new_superclass,
                        }
                    )
                    symmetric_noisy_classes_simple.add(
                        (int(original_label), int(new_label))
                    )

        D_tr_data = np.concatenate([D_f_data, D_n_data], axis=0)
        D_tr_labels = np.concatenate([D_f_labels, D_n_labels], axis=0)

        # 打乱训练数据集
        perm = rng.permutation(len(D_tr_data))
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

        # 保存训练数据集
        train_data_path = settings.get_dataset_path(
            dataset_name, case, "train_data", t + 1
        )
        train_label_path = settings.get_dataset_path(
            dataset_name, case, "train_label", t + 1
        )

        subdir = os.path.dirname(train_data_path)
        os.makedirs(subdir, exist_ok=True)

        np.save(train_data_path, D_tr_data)
        np.save(train_label_path, D_tr_labels)

        print(f"D_tr 版本 {t+1} 已保存到 {subdir}")

    # 记录噪声注入信息
    if noise_type == "symmetric":
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_symmetric_noisy_classes_detailed.json",
            "w",
        ) as f:
            json.dump(symmetric_noisy_classes, f, indent=4)
        with open(
            f"{dataset_name}_{noise_type}_{noise_ratio}_symmetric_noisy_classes_simple.json",
            "w",
        ) as f:
            json.dump(list(symmetric_noisy_classes_simple), f, indent=4)

    print("所有数据集生成完毕。")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate FLOWER-102 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/flower-102/normal",
        help="原始 FLOWER-102 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/flower-102/gen/",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric"],
        default="symmetric",
        help="标签噪声类型：'symmetric'",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="噪声比例（默认 0.2）"
    )
    parser.add_argument(
        "--num_versions", type=int, default=3, help="生成的增量版本数量（默认 3）"
    )
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.1],
        help="各增量版本的保留比例列表",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="是否使用类均衡的数据划分方式。如果不指定，则使用随机划分。",
    )

    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.5,
        help="类均衡划分的比例，默认为 0.5",
    )

    args = parser.parse_args()

    create_flower102_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
        balanced=args.balanced,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()

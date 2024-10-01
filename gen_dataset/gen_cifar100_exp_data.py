import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
import json

import collections


def split_by_class(data, labels, num_classes=100):
    """按类别划分数据集"""
    class_data = {i: [] for i in range(num_classes)}
    for i, label in enumerate(labels):
        class_data[label.item()].append(data[i])
    return class_data


def sample_class_balanced_data(class_data, split_ratio=0.5):
    """按比例从每个类别中均衡抽取样本"""
    D_0_data = []
    D_0_labels = []
    D_inc_data = []
    D_inc_labels = []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        split_idx = int(num_samples * split_ratio)

        # 打乱样本
        shuffled_indices = np.random.permutation(num_samples)

        # D_0 获取前半部分数据
        D_0_data.extend([samples[i] for i in shuffled_indices[:split_idx]])
        D_0_labels.extend([class_label] * split_idx)

        # D_inc 获取后半部分数据
        D_inc_data.extend([samples[i] for i in shuffled_indices[split_idx:]])
        D_inc_labels.extend([class_label] * (num_samples - split_idx))

    # D_0_data = torch.stack(D_0_data)
    # D_0_labels = torch.tensor(D_0_labels)
    # D_inc_data = torch.stack(D_inc_data)
    # D_inc_labels = torch.tensor(D_inc_labels)

    D_0_data = np.stack(D_0_data)
    D_0_labels = np.array(D_0_labels)
    D_inc_data = np.stack(D_inc_data)
    D_inc_labels = np.array(D_inc_labels)

    return D_0_data, D_0_labels, D_inc_data, D_inc_labels


def sample_replay_data(D_0_data, D_0_labels, replay_ratio=0.1):
    """从 D_0 中均衡抽取样本作为重放数据集 D_a"""
    class_data = split_by_class(D_0_data, D_0_labels)
    D_a_data = []
    D_a_labels = []

    for class_label, samples in class_data.items():
        num_samples = len(samples)
        num_replay_samples = int(num_samples * replay_ratio)
        replay_indices = np.random.choice(
            num_samples, num_replay_samples, replace=False
        )

        D_a_data.extend([samples[i] for i in replay_indices])
        D_a_labels.extend([class_label] * num_replay_samples)

    # D_a_data = torch.stack(D_a_data)
    # D_a_labels = torch.tensor(D_a_labels)

    D_a_data = np.stack(D_a_data)
    D_a_labels = np.array(D_a_labels)

    return D_a_data, D_a_labels


def load_classes_from_file(file_path):
    """从文件中读取类别列表"""
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_cifar100_superclass_mapping(file_path):
    """从JSON文件中加载 CIFAR-100 的 superclass 与 child class 的映射"""
    with open(file_path, "r") as f:
        cifar100_superclass_to_child = json.load(f)
    return cifar100_superclass_to_child


def build_asymmetric_mapping(superclass_mapping, classes):
    """构建非对称标签映射，确保标签替换为同superclass内的其他类"""
    child_to_superclass_mapping = {}

    # 构建child class到superclass的反向映射
    for superclass, child_classes in superclass_mapping.items():
        for child_class in child_classes:
            child_to_superclass_mapping[child_class] = (superclass, child_classes)

    # 构建非对称映射表
    asymmetric_mapping = {}
    for class_name in classes:
        # 获取该类别所属的大类（superclass）以及该大类中的所有类别
        if class_name in child_to_superclass_mapping:
            superclass, child_classes = child_to_superclass_mapping[class_name]
            # 在同一superclass中随机选择一个不同的类别作为替换
            available_classes = [c for c in child_classes if c != class_name]
            if available_classes:
                new_class = np.random.choice(available_classes)
                asymmetric_mapping[class_name] = new_class
            else:
                asymmetric_mapping[class_name] = (
                    class_name  # 如果没有其他类别，则保持原标签不变
                )
    return asymmetric_mapping


def create_cifar100_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
    balanced=False,
):
    # transform = transforms.Compose([transforms.ToTensor()])

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为张量，并归一化到 [0, 1]
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),  # 标准化
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    # 加载 CIFAR-100 数据集
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(
        [train_dataset[i][1] for i in range(len(train_dataset))]
    )
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # 根据 balanced 参数调整存储路径
    if balanced:
        print("使用类均衡的数据划分方式...")
        subdir = os.path.join(gen_dir, f"nr_{noise_ratio}_nt_{noise_type}_balanced")
        # 按类别划分训练数据
        class_data = split_by_class(train_data, train_labels)

        # 构建类均衡的 D_0 和 D_inc_0
        D_0_data, D_0_labels, D_inc_data, D_inc_labels = sample_class_balanced_data(
            class_data, split_ratio=0.5
        )

        # 构建重放数据集 D_a（从 D_0 中随机抽取 10% 的样本）
        D_a_data, D_a_labels = sample_replay_data(
            D_0_data, D_0_labels, replay_ratio=0.1
        )

    else:
        print("使用随机的数据划分方式...")
        subdir = os.path.join(gen_dir, f"nr_{noise_ratio}_nt_{noise_type}")
        # 随机划分初始数据集 D_0 和增量数据集 D_inc^(0)
        num_samples = len(train_data)
        indices = np.random.permutation(num_samples)
        split_idx = num_samples // 2

        D_0_indices = indices[:split_idx]
        D_inc_indices = indices[split_idx:]

        D_0_data = train_data[D_0_indices]
        D_0_labels = train_labels[D_0_indices]

        D_inc_data = train_data[D_inc_indices]
        D_inc_labels = train_labels[D_inc_indices]

        # 构建重放数据集 D_a（从 D_0 中随机抽取 10% 的样本）
        num_replay_samples = int(len(D_0_data) * 0.1)
        D_a_indices = np.random.choice(len(D_0_data), num_replay_samples, replace=False)
        D_a_data = D_0_data[D_a_indices]
        D_a_labels = D_0_labels[D_a_indices]

    # 创建存储目录
    os.makedirs(subdir, exist_ok=True)

    # 检查 D_0 数据分布
    print("D_0 Labels distribution:", collections.Counter(D_0_labels))

    # 检查 D_inc 数据分布
    print("D_inc Labels distribution:", collections.Counter(D_inc_labels))


    # 保存初始数据集、初始增量数据集、重放数据集
    np.save(os.path.join(subdir, "D_0.npy"), D_0_data)
    np.save(os.path.join(subdir, "D_0_labels.npy"), D_0_labels)

    np.save(os.path.join(subdir, "D_inc_0_data.npy"), D_inc_data)
    np.save(os.path.join(subdir, "D_inc_0_labels.npy"), D_inc_labels)

    np.save(os.path.join(subdir, "D_a.npy"), D_a_data)
    np.save(os.path.join(subdir, "D_a_labels.npy"), D_a_labels)

    # 保存训练数据集
    np.save(os.path.join(subdir, "train_data.npy"), train_data)
    np.save(os.path.join(subdir, "train_labels.npy"), train_labels)

    # 保存测试数据集
    np.save(os.path.join(subdir, "test_data.npy"), test_data)
    np.save(os.path.join(subdir, "test_labels.npy"), test_labels)

    num_classes = 100

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

    # 定义非对称噪声映射

    # 读取 CIFAR-100 类别
    cifar100_classes_file = "./configs/classes/cifar_100_classes.txt"
    cifar100_classes = load_classes_from_file(cifar100_classes_file)

    # 读取 CIFAR-100 的 superclass 和 child class 映射
    cifar100_mapping_file = "./configs/classes/cifar_100_mapping.json"
    cifar100_superclass_mapping = load_cifar100_superclass_mapping(
        cifar100_mapping_file
    )

    # 打印读取到的类别信息
    print("CIFAR-100 Classes:", cifar100_classes)

    # 构建非对称映射，如果选择了非对称噪声
    if noise_type == "asymmetric":
        asymmetric_mapping = build_asymmetric_mapping(
            cifar100_superclass_mapping, cifar100_classes
        )

    # 生成增量版本数据集
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # 模拟遗忘：根据保留比例抽取遗忘类别的样本
        num_forget_samples = int(len(D_inc_forget_indices) * retention_ratio)

        if num_forget_samples > 0:
            forget_sample_indices = np.random.choice(
                D_inc_forget_indices, num_forget_samples, replace=False
            )
            D_f_data = D_inc_data[forget_sample_indices]
            D_f_labels = D_inc_labels[forget_sample_indices]
        else:
            D_f_data = torch.empty(0, 3, 32, 32)
            D_f_labels = torch.empty(0, dtype=torch.long)

        # 噪声注入：对噪声类别的样本注入噪声
        noise_sample_indices = D_inc_noise_indices
        num_noisy_samples = int(len(noise_sample_indices) * noise_ratio)

        if num_noisy_samples > 0:
            noisy_indices = np.random.choice(
                noise_sample_indices, num_noisy_samples, replace=False
            )
        else:
            noisy_indices = []

        D_n_data = D_inc_data[noise_sample_indices]
        # D_n_labels = D_inc_labels[noise_sample_indices].clone()
        D_n_labels = D_inc_labels[noise_sample_indices].copy()

        # 在 D_n_labels 中注入噪声
        for idx_in_D_n, D_inc_idx in enumerate(noise_sample_indices):
            if D_inc_idx in noisy_indices:
                original_label = D_n_labels[idx_in_D_n].item()
                original_class_name = cifar100_classes[original_label]

                if noise_type == "symmetric":
                    new_label = original_label
                    while new_label == original_label:
                        new_label = np.random.randint(0, num_classes)
                    D_n_labels[idx_in_D_n] = new_label
                elif noise_type == "asymmetric":
                    if original_class_name in asymmetric_mapping:
                        new_class_name = asymmetric_mapping[original_class_name]
                        new_label = cifar100_classes.index(new_class_name)
                        D_n_labels[idx_in_D_n] = new_label
                else:
                    raise ValueError("Invalid noise type.")
            else:
                # 未被选中注入噪声的样本标签保持不变
                pass

        # D_tr_data = torch.cat([D_f_data, D_n_data], dim=0)
        # D_tr_labels = torch.cat([D_f_labels, D_n_labels], dim=0)

        D_tr_data = np.concatenate([D_f_data, D_n_data], axis=0)
        D_tr_labels = np.concatenate([D_f_labels, D_n_labels], axis=0)

        # 打乱训练数据集
        # perm = torch.randperm(len(D_tr_data))
        perm = np.random.permutation(len(D_tr_data))  # 使用 numpy 的随机打乱
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

        # 保存训练数据集
        # torch.save(D_tr_data, os.path.join(subdir, f"D_tr_data_version_{t+1}.npy"))
        # torch.save(D_tr_labels, os.path.join(subdir, f"D_tr_labels_version_{t+1}.npy"))

        # 保存训练数据集
        np.save(os.path.join(subdir, f"D_tr_data_version_{t+1}.npy"), D_tr_data)
        np.save(os.path.join(subdir, f"D_tr_labels_version_{t+1}.npy"), D_tr_labels)

        print(f"D_tr 版本 {t+1} 已保存到 {subdir}")

    print("所有数据集生成完毕。")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-100 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-100/normal",
        help="原始 CIFAR-100 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-100/gen/",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="标签噪声类型：'symmetric' 或 'asymmetric'",
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

    args = parser.parse_args()

    create_cifar100_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
        balanced=args.balanced,
    )


if __name__ == "__main__":
    main()

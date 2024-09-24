import torch
import numpy as np
import os
from torchvision import datasets, transforms


def create_cifar10_npy_files(
    data_dir,
    noise_dir,
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
):
    """
    基于CIFAR-10数据集，生成带有噪声的训练、辅助和测试数据集，并保存为.npy文件到指定目录。
    :param data_dir: 原始CIFAR-10数据集的目录
    :param noise_dir: 噪声数据集的保存目录
    :param noise_ratio: 增量数据集中的噪声比例
    :param num_versions: 生成的增量版本数量
    :param retention_ratios: 各增量版本的Retention ratio列表
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载CIFAR-10训练和测试数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # 提取训练集和测试集数据与标签
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(
        [train_dataset[i][1] for i in range(len(train_dataset))]
    )

    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # Step 1: 从 50000 样本中随机划分 50% 样本为 D_0（25000 样本）
    num_samples = len(train_data)
    indices = np.random.permutation(num_samples)
    split_idx = num_samples // 2
    D_0_indices = indices[:split_idx]
    D_inc_indices = indices[split_idx:]

    D_0_data = train_data[D_0_indices]
    D_0_labels = train_labels[D_0_indices]

    # 定义遗忘类和非遗忘类
    forget_classes = [1, 3, 5, 7, 9]
    non_forget_classes = [0, 2, 4, 6, 8]

    # 将 D_0 分为遗忘类和非遗忘类
    D_0_forget_indices = [
        i for i in range(len(D_0_labels)) if D_0_labels[i] in forget_classes
    ]
    D_0_non_forget_indices = [
        i for i in range(len(D_0_labels)) if D_0_labels[i] in non_forget_classes
    ]

    D_0_forget_data = D_0_data[D_0_forget_indices]
    D_0_forget_labels = D_0_labels[D_0_forget_indices]
    D_0_non_forget_data = D_0_data[D_0_non_forget_indices]
    D_0_non_forget_labels = D_0_labels[D_0_non_forget_indices]

    # 生成重放数据 D_a (10%)
    num_replay_samples = int(len(D_0_data) * 0.1)
    D_a_indices = np.random.choice(split_idx, num_replay_samples, replace=False)
    D_a_data = D_0_data[D_a_indices]
    D_a_labels = D_0_labels[D_a_indices]

    # 生成增量数据集 D_inc 的不同版本
    D_inc_versions = []
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # 从遗忘类和非遗忘类中分别抽取样本
        num_forget_samples = int(len(D_0_forget_data) * retention_ratio * 0.5)
        num_non_forget_samples = int(len(D_0_non_forget_data) * retention_ratio * 0.5)

        if num_forget_samples > 0:
            D_inc_forget_indices = np.random.choice(
                len(D_0_forget_data), num_forget_samples, replace=False
            )
            D_inc_forget_data = D_0_forget_data[D_inc_forget_indices]
            D_inc_forget_labels = D_0_forget_labels[D_inc_forget_indices]
        else:
            D_inc_forget_data = torch.empty(0, 3, 32, 32)
            D_inc_forget_labels = torch.empty(0, dtype=torch.long)

        if num_non_forget_samples > 0:
            D_inc_non_forget_indices = np.random.choice(
                len(D_0_non_forget_data), num_non_forget_samples, replace=False
            )
            D_inc_non_forget_data = D_0_non_forget_data[D_inc_non_forget_indices]
            D_inc_non_forget_labels = D_0_non_forget_labels[D_inc_non_forget_indices]
        else:
            D_inc_non_forget_data = torch.empty(0, 3, 32, 32)
            D_inc_non_forget_labels = torch.empty(0, dtype=torch.long)

        # 合并数据
        D_inc_data = torch.cat([D_inc_forget_data, D_inc_non_forget_data], dim=0)
        D_inc_labels = torch.cat([D_inc_forget_labels, D_inc_non_forget_labels], dim=0)

        # 添加噪声标签到非遗忘类
        num_noisy_samples = int(len(D_inc_non_forget_data) * noise_ratio)
        if num_noisy_samples > 0:
            noise_indices = np.random.choice(
                num_non_forget_samples, num_noisy_samples, replace=False
            )
            for idx in noise_indices:
                original_label = D_inc_labels[num_forget_samples + idx].item()
                noisy_label = original_label
                while noisy_label == original_label:
                    noisy_label = np.random.randint(0, 10)
                D_inc_labels[num_forget_samples + idx] = noisy_label

        D_inc_versions.append((D_inc_data, D_inc_labels))

    # 保存数据集
    os.makedirs(noise_dir, exist_ok=True)
    torch.save(D_0_data, os.path.join(noise_dir, "D_0.npy"))
    torch.save(D_0_labels, os.path.join(noise_dir, "D_0_labels.npy"))
    torch.save(D_a_data, os.path.join(noise_dir, "D_a.npy"))
    torch.save(D_a_labels, os.path.join(noise_dir, "D_a_labels.npy"))

    for t, (data, labels) in enumerate(D_inc_versions):
        torch.save(data, os.path.join(noise_dir, f"D_inc_{t+1}.npy"))
        torch.save(labels, os.path.join(noise_dir, f"D_inc_labels_{t+1}.npy"))

    # 保存测试集
    torch.save(test_data, os.path.join(noise_dir, "test_data.npy"))
    torch.save(test_labels, os.path.join(noise_dir, "test_labels.npy"))

    return (
        train_data[D_inc_indices],
        train_labels[D_inc_indices],
        test_data,
        test_labels,
    )


def create_incremental_data_versions(
    D_a_data, D_a_labels, D_inc_versions, save_dir=None
):
    """
    构建多个增量训练数据 D_tr 版本，每个版本是 D_a 和对应的 D_inc_{version_number} 的合集。
    :param D_a_data: 重放数据集 D_a 的数据部分
    :param D_a_labels: 重放数据集 D_a 的标签部分
    :param D_inc_versions: 一个包含多个版本的增量数据集 D_inc
    :param save_dir: 保存生成的 D_tr 数据集的路径
    """
    for version_num, (D_inc_data, D_inc_labels) in enumerate(D_inc_versions, start=1):
        D_tr_data = torch.cat([D_a_data, D_inc_data], dim=0)
        D_tr_labels = torch.cat([D_a_labels, D_inc_labels], dim=0)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(
                os.path.join(save_dir, f"cifar10_D_tr_data_version_{version_num}.npy"),
                D_tr_data.numpy(),
            )
            np.save(
                os.path.join(
                    save_dir, f"cifar10_D_tr_labels_version_{version_num}.npy"
                ),
                D_tr_labels.numpy(),
            )
            print(f"D_tr 版本 {version_num} 已保存到 {save_dir}")

    print("所有 D_tr 版本数据保存完毕！")


# 主程序
if __name__ == "__main__":
    data_dir = "./data/cifar-10"
    noise_data_dir = os.path.join(data_dir, "noise")  # 存储增量数据的路径

    # Step 1: 创建 D_a 和多个 D_inc 版本
    train_data_Dinc, train_labels_Dinc, test_data, test_labels = (
        create_cifar10_npy_files(data_dir, noise_data_dir)
    )

    # 从上一步生成的版本中加载 D_a 和 D_inc 数据
    D_a_data = torch.load(os.path.join(noise_data_dir, "D_a.npy"))
    D_a_labels = torch.load(os.path.join(noise_data_dir, "D_a_labels.npy"))

    # 加载多个增量版本的数据和标签
    D_inc_versions = []
    for t in range(3):  # 假设有3个增量版本
        D_inc_data = torch.load(os.path.join(noise_data_dir, f"D_inc_{t+1}.npy"))
        D_inc_labels = torch.load(
            os.path.join(noise_data_dir, f"D_inc_labels_{t+1}.npy")
        )
        D_inc_versions.append((D_inc_data, D_inc_labels))

    # Step 2: 创建并保存每个版本的 D_tr
    create_incremental_data_versions(
        D_a_data,
        D_a_labels,
        D_inc_versions,
        save_dir=noise_data_dir,
    )

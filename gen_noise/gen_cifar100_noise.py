import numpy as np
import os
import torch
from torchvision import datasets, transforms


def create_cifar100_npy_files(
    data_dir, noise_dir, noise_ratio=0.2, forget_class_ratio=0.05
):
    """
    基于CIFAR-100数据集，生成带有噪声的训练、辅助和测试数据集，并保存为.npy文件到指定目录。
    :param data_dir: 原始CIFAR-100数据集的目录
    :param noise_dir: 噪声数据集的保存目录
    :param noise_ratio: 增量数据集中的噪声比例
    :param forget_class_ratio: 遗忘类的比例
    """

    # 定义CIFAR-100的图像转换
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载CIFAR-100训练和测试数据集
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    # 提取训练集和测试集数据与标签
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(
        [train_dataset[i][1] for i in range(len(train_dataset))]
    )

    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # Step 1: 随机划分 D_0 和 D_inc，分别包含 25000 个样本
    total_train_indices = torch.randperm(len(train_data))

    D_0_indices = total_train_indices[:25000]  # D_0 占 50%
    D_inc_indices = total_train_indices[25000:]  # D_inc 占 50%

    train_data_D0, train_labels_D0 = train_data[D_0_indices], train_labels[D_0_indices]
    train_data_Dinc, train_labels_Dinc = (
        train_data[D_inc_indices],
        train_labels[D_inc_indices],
    )

    # Step 2: 从 D_0 中抽取 10% 样本作为 replay 数据集 D_a (2500 个样本)
    aux_size = 2500
    aux_indices = torch.randperm(len(train_data_D0))[:aux_size]
    aux_data, aux_labels = train_data_D0[aux_indices], train_labels_D0[aux_indices]

    # **注意：D_0 保持不变**

    # Step 3: 构造 D_tr：从 D_inc 中构建增量数据集 D_tr，包含噪声和遗忘类
    class_labels = torch.unique(train_labels_Dinc)

    # 获取需要遗忘的类
    forget_class_count = int(forget_class_ratio * len(class_labels))
    forget_classes = class_labels[
        torch.randperm(len(class_labels))[:forget_class_count]
    ]

    # 标记每个类的数据
    forget_class_mask = torch.isin(train_labels_Dinc, forget_classes)
    other_class_mask = ~forget_class_mask

    # 遗忘类的数据抽取
    forget_class_data = train_data_Dinc[forget_class_mask]
    forget_class_labels = train_labels_Dinc[forget_class_mask]

    # 剩余类别数据抽取，并添加20%噪声
    other_class_data = train_data_Dinc[other_class_mask]
    other_class_labels = train_labels_Dinc[other_class_mask]

    # 在非遗忘类中添加20%的噪声
    noisy_labels = other_class_labels.clone()
    noise_indices = torch.randperm(len(noisy_labels))[
        : int(noise_ratio * len(noisy_labels))
    ]
    noisy_labels[noise_indices] = torch.randint(
        0, 100, size=(len(noise_indices),)
    )  # CIFAR-100 标签范围为 0-99

    # 构建 D_tr
    D_tr_data = torch.cat([forget_class_data, other_class_data], dim=0)
    D_tr_labels = torch.cat([forget_class_labels, noisy_labels], dim=0)

    # 打乱 D_tr 的顺序
    D_tr_indices = torch.randperm(len(D_tr_data))
    D_tr_data = D_tr_data[D_tr_indices]
    D_tr_labels = D_tr_labels[D_tr_indices]

    # 确保噪声保存目录存在
    os.makedirs(noise_dir, exist_ok=True)

    # 保存生成的数据到 noise 目录
    # 保存训练集 D_0
    np.save(f"{noise_dir}/cifar100_train_data.npy", train_data_D0.numpy())
    np.save(f"{noise_dir}/cifar100_train_labels.npy", train_labels_D0.numpy())

    # 保存辅助集 D_a
    np.save(f"{noise_dir}/cifar100_aux_data.npy", aux_data.numpy())
    np.save(f"{noise_dir}/cifar100_aux_labels.npy", aux_labels.numpy())

    # 保存增量数据集 D_tr
    np.save(f"{noise_dir}/cifar100_inc_data.npy", D_tr_data.numpy())
    np.save(f"{noise_dir}/cifar100_inc_labels.npy", D_tr_labels.numpy())

    # 保存测试集 D_ts
    np.save(f"{noise_dir}/cifar100_test_data.npy", test_data.numpy())
    np.save(f"{noise_dir}/cifar100_test_labels.npy", test_labels.numpy())

    print(f"CIFAR-100 数据集已保存为 .npy 文件到 {noise_dir}")


if __name__ == "__main__":
    data_dir = "./data/cifar-100"  # CIFAR-100 原始数据集的路径
    noise_dir = os.path.join(data_dir, "noise")  # 存储带噪声数据集的路径
    create_cifar100_npy_files(data_dir, noise_dir)

    # 加载生成的 npy 文件
    train_data = np.load("data/cifar-100/noise/cifar100_train_data.npy")
    train_labels = np.load("data/cifar-100/noise/cifar100_train_labels.npy")

    aux_data = np.load("data/cifar-100/noise/cifar100_aux_data.npy")
    aux_labels = np.load("data/cifar-100/noise/cifar100_aux_labels.npy")

    inc_data = np.load("data/cifar-100/noise/cifar100_inc_data.npy")
    inc_labels = np.load("data/cifar-100/noise/cifar100_inc_labels.npy")

    test_data = np.load("data/cifar-100/noise/cifar100_test_data.npy")
    test_labels = np.load("data/cifar-100/noise/cifar100_test_labels.npy")

    # 检查加载的数据形状是否与预期一致
    print(
        f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}"
    )
    print(f"Aux data shape: {aux_data.shape}, Aux labels shape: {aux_labels.shape}")
    print(
        f"Incremental data shape: {inc_data.shape}, Incremental labels shape: {inc_labels.shape}"
    )
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

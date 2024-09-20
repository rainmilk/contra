import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import os
from noise_datasets import NoiseCIFAR10  # 导入 noise_dataset 中的噪声处理


# Step 1: 生成增量数据集D_tr的代码
def create_cifar10_npy_files(
    data_dir, noise_dir, noise_ratio=0.2, forget_class_ratio=0.05
):
    """
    基于CIFAR-10数据集，生成带有噪声的训练、辅助和测试数据集，并保存为.npy文件到指定目录。
    :param data_dir: 原始CIFAR-10数据集的目录
    :param noise_dir: 噪声数据集的保存目录
    :param noise_ratio: 增量数据集中的噪声比例
    :param forget_class_ratio: 遗忘类的比例
    """

    # 加载CIFAR-10训练和测试数据集
    train_dataset = NoiseCIFAR10(
        root=data_dir, train=True, download=True, noise_type="sym", percent=noise_ratio
    )
    test_dataset = NoiseCIFAR10(
        root=data_dir, train=False, download=True, noise_type="none"
    )

    # 提取训练集和测试集数据与标签
    train_data = torch.tensor(train_dataset.data).permute(
        0, 3, 1, 2
    )  # 转换为通道优先格式
    train_labels = torch.tensor(train_dataset.targets)

    test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
    test_labels = torch.tensor(test_dataset.targets)

    # Step 1: 从 50000 样本中随机划分 50% 样本为 D_0（25000 样本），剩余 50% 样本为 D_inc（25000 样本）
    total_indices = torch.randperm(len(train_data))
    D_0_indices = total_indices[:25000]
    D_inc_indices = total_indices[25000:]

    train_data_D0, train_labels_D0 = train_data[D_0_indices], train_labels[D_0_indices]
    train_data_Dinc, train_labels_Dinc = (
        train_data[D_inc_indices],
        train_labels[D_inc_indices],
    )

    # Step 2: 从 D_0 中抽取 10% 样本作为 replay 数据集 D_a (2500 个样本)
    aux_size = 2500
    aux_indices = torch.randperm(len(train_data_D0))[:aux_size]
    aux_data, aux_labels = train_data_D0[aux_indices], train_labels_D0[aux_indices]

    # **注意：D_0 保持不变，不应移除 aux 数据**
    # 不再需要删除 `aux_indices`，`train_data_D0` 应该保持完整的 25000 样本。

    # 保存生成的数据到noise目录
    os.makedirs(noise_dir, exist_ok=True)

    # 保存训练集 D_0
    np.save(f"{noise_dir}/cifar10_train_data.npy", train_data_D0.numpy())
    np.save(f"{noise_dir}/cifar10_train_labels.npy", train_labels_D0.numpy())

    # 保存辅助集 D_a
    np.save(f"{noise_dir}/cifar10_aux_data.npy", aux_data.numpy())
    np.save(f"{noise_dir}/cifar10_aux_labels.npy", aux_labels.numpy())

    # 保存测试集 D_ts
    np.save(f"{noise_dir}/cifar10_test_data.npy", test_data.numpy())
    np.save(f"{noise_dir}/cifar10_test_labels.npy", test_labels.numpy())

    return train_data_Dinc, train_labels_Dinc


# Step 2: 训练增量学习模型
def create_incremental_data(
    train_data_Dinc, train_labels_Dinc, forget_classes, noise_ratio=0.2
):
    """
    从 D_inc 中构造增量训练数据 D_tr。
    对遗忘类仅采样10%数据，非遗忘类采样50%数据，并对其中20%样本添加标签噪声。
    """
    # 标记每个类的数据
    forget_class_mask = torch.isin(train_labels_Dinc, forget_classes)
    other_class_mask = ~forget_class_mask

    # 遗忘类的数据抽取（每类采样10%）
    forget_class_data = train_data_Dinc[forget_class_mask]
    forget_class_labels = train_labels_Dinc[forget_class_mask]
    forget_sample_indices = torch.randperm(len(forget_class_data))[
        : int(0.1 * len(forget_class_data))
    ]
    forget_class_data = forget_class_data[forget_sample_indices]
    forget_class_labels = forget_class_labels[forget_sample_indices]

    # 非遗忘类数据抽取（每类采样50%）
    other_class_data = train_data_Dinc[other_class_mask]
    other_class_labels = train_labels_Dinc[other_class_mask]
    other_sample_indices = torch.randperm(len(other_class_data))[
        : int(0.5 * len(other_class_data))
    ]
    sampled_other_class_data = other_class_data[other_sample_indices]
    sampled_other_class_labels = other_class_labels[other_sample_indices]

    # 对采样的非遗忘类数据中的20%进行标签噪声处理
    # 使用 NoiseCIFAR10 类进行处理
    noise_dataset = NoiseCIFAR10(
        root=".", train=True, noise_type="sym", percent=noise_ratio
    )
    noisy_labels = torch.tensor(noise_dataset.targets)

    # 构建增量训练数据 D_tr
    D_tr_data = torch.cat([forget_class_data, sampled_other_class_data], dim=0)
    D_tr_labels = torch.cat(
        [forget_class_labels, noisy_labels[: len(sampled_other_class_labels)]], dim=0
    )

    return D_tr_data, D_tr_labels


class IncrementalLearningModel(nn.Module):
    def __init__(self, num_classes=10):
        super(IncrementalLearningModel, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_incremental_model(
    D_tr_data, D_tr_labels, model, num_epochs=10, batch_size=64, lr=0.001
):
    """
    训练增量学习模型 M_p(D_tr)。
    :param D_tr_data: 增量数据集 D_tr 的输入数据
    :param D_tr_labels: 增量数据集 D_tr 的标签
    :param model: 增量学习模型
    :param num_epochs: 训练的迭代次数
    :param batch_size: 每个batch的样本数
    :param lr: 学习率
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(D_tr_data, D_tr_labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    print("Incremental model training complete.")
    return model


def validate_npy_files(noise_dir):
    """
    校验生成的npy文件是否正确，主要检查文件的形状。
    :param noise_dir: 噪声数据集的保存路径
    """
    # 加载生成的 npy 文件
    train_data = np.load(f"{noise_dir}/cifar10_train_data.npy")
    train_labels = np.load(f"{noise_dir}/cifar10_train_labels.npy")

    aux_data = np.load(f"{noise_dir}/cifar10_aux_data.npy")
    aux_labels = np.load(f"{noise_dir}/cifar10_aux_labels.npy")

    test_data = np.load(f"{noise_dir}/cifar10_test_data.npy")
    test_labels = np.load(f"{noise_dir}/cifar10_test_labels.npy")

    # 检查加载的数据形状是否与预期一致
    assert train_data.shape == (
        25000,
        3,
        32,
        32,
    ), f"Train data shape mismatch: {train_data.shape}"
    assert train_labels.shape == (
        25000,
    ), f"Train labels shape mismatch: {train_labels.shape}"

    assert aux_data.shape == (
        2500,
        3,
        32,
        32,
    ), f"Aux data shape mismatch: {aux_data.shape}"
    assert aux_labels.shape == (2500,), f"Aux labels shape mismatch: {aux_labels.shape}"

    assert test_data.shape == (
        10000,
        3,
        32,
        32,
    ), f"Test data shape mismatch: {test_data.shape}"
    assert test_labels.shape == (
        10000,
    ), f"Test labels shape mismatch: {test_labels.shape}"

    print("All .npy files are correctly generated and validated.")


if __name__ == "__main__":
    data_dir = "./data/cifar-10"  # CIFAR-10 原始数据集的路径
    noise_dir = os.path.join(data_dir, "noise")  # 存储带噪声数据集的路径

    # Step 1: 创建 D_0, D_inc 和 测试集数据
    train_data_Dinc, train_labels_Dinc = create_cifar10_npy_files(data_dir, noise_dir)

    # Step 2: 校验生成的npy文件
    validate_npy_files(noise_dir)

    # Step 3: 构建增量数据集 D_tr
    forget_classes = torch.tensor([1, 3, 5, 7, 9])  # 假设选择这5个类作为遗忘类
    D_tr_data, D_tr_labels = create_incremental_data(
        train_data_Dinc, train_labels_Dinc, forget_classes
    )

    # Step 4: 定义并训练增量学习模型
    incremental_model = IncrementalLearningModel(num_classes=10)
    trained_model = train_incremental_model(D_tr_data, D_tr_labels, incremental_model)

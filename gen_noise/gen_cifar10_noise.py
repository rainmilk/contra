import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os


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

    return train_data_Dinc, train_labels_Dinc, test_data, test_labels


def test_model(model, test_data, test_labels, batch_size=64, device="cpu"):
    """
    在测试集上评估模型的准确率。
    :param model: 已训练的模型
    :param test_data: 测试集数据
    :param test_labels: 测试集标签
    :param batch_size: 每个 batch 的大小
    :return: 测试集准确率
    """
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU上
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# Step 2: 训练增量学习模型
def create_incremental_data(
    train_data_Dinc, train_labels_Dinc, forget_classes, noise_ratio=0.2, save_dir=None
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
    noisy_labels = sampled_other_class_labels.clone()
    noise_indices = torch.randperm(len(noisy_labels))[
        : int(noise_ratio * len(noisy_labels))
    ]
    noisy_labels[noise_indices] = torch.randint(0, 10, size=(len(noise_indices),))

    # 构建增量训练数据 D_tr
    D_tr_data = torch.cat([forget_class_data, sampled_other_class_data], dim=0)
    D_tr_labels = torch.cat([forget_class_labels, noisy_labels], dim=0)

    # 保存增量数据 D_tr
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/cifar10_D_tr_data.npy", D_tr_data.numpy())
        np.save(f"{save_dir}/cifar10_D_tr_labels.npy", D_tr_labels.numpy())
        print(f"D_tr 数据已保存到 {save_dir}")

    return D_tr_data, D_tr_labels


def validate_d_tr(save_dir):
    """
    校验生成的 D_tr 数据集文件是否正确。
    """
    D_tr_data = np.load(f"{save_dir}/cifar10_D_tr_data.npy")
    D_tr_labels = np.load(f"{save_dir}/cifar10_D_tr_labels.npy")

    # 检查加载的数据形状是否合理
    print(f"D_tr_data shape: {D_tr_data.shape}")
    print(f"D_tr_labels shape: {D_tr_labels.shape}")
    assert D_tr_data.shape[1:] == (
        3,
        32,
        32,
    ), f"D_tr_data shape mismatch: {D_tr_data.shape}"
    assert (
        len(D_tr_labels.shape) == 1
    ), f"D_tr_labels shape mismatch: {D_tr_labels.shape}"

    print("D_tr 数据校验通过！")


class IncrementalLearningModel(nn.Module):
    def __init__(self, num_classes=10):
        super(IncrementalLearningModel, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_models(
    D_tr_data,
    D_tr_labels,
    original_train_data,
    original_train_labels,
    test_data,
    test_labels,
    incremental_model,
    original_model,
    num_epochs=100,
    batch_size=64,
    lr=0.001,
    incremental_save_path="incremental_model.pth",
    original_save_path="original_model.pth",
    device="cpu",
):
    """
    同时训练两个模型：一个在增量数据集上，一个在原始CIFAR-10数据集上。
    """
    # 将模型移到设备上
    incremental_model = incremental_model.to(device)
    original_model = original_model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_incremental = optim.Adam(incremental_model.parameters(), lr=lr)
    optimizer_original = optim.Adam(original_model.parameters(), lr=lr)

    # 创建数据加载器
    incremental_dataset = torch.utils.data.TensorDataset(D_tr_data, D_tr_labels)
    original_dataset = torch.utils.data.TensorDataset(
        original_train_data, original_train_labels
    )

    incremental_loader = torch.utils.data.DataLoader(
        incremental_dataset, batch_size=batch_size, shuffle=True
    )
    original_loader = torch.utils.data.DataLoader(
        original_dataset, batch_size=batch_size, shuffle=True
    )

    # 训练模型
    for epoch in range(num_epochs):
        incremental_model.train()  # 将增量模型设置为训练模式
        original_model.train()  # 将原始模型设置为训练模式

        incremental_loss = 0.0
        original_loss = 0.0

        # 训练增量学习模型
        for inputs, labels in incremental_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU上
            optimizer_incremental.zero_grad()
            outputs = incremental_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_incremental.step()
            incremental_loss += loss.item()

        # 训练原始CIFAR-10模型
        for inputs, labels in original_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU上
            optimizer_original.zero_grad()
            outputs = original_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_original.step()
            original_loss += loss.item()

        # 在测试集上评估两者性能
        incremental_test_accuracy = test_model(
            incremental_model, test_data, test_labels, batch_size, device
        )
        original_test_accuracy = test_model(
            original_model, test_data, test_labels, batch_size, device
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Incremental Model - Loss: {incremental_loss / len(incremental_loader)}, Test Accuracy: {incremental_test_accuracy}"
        )
        print(
            f"Original Model - Loss: {original_loss / len(original_loader)}, Test Accuracy: {original_test_accuracy}"
        )

    # 保存训练后的模型
    torch.save(incremental_model.state_dict(), incremental_save_path)
    torch.save(original_model.state_dict(), original_save_path)
    print(f"Incremental Model saved to {incremental_save_path}")
    print(f"Original Model saved to {original_save_path}")

    return incremental_model, original_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "./data/cifar-10"  # CIFAR-10 原始数据集的路径
    noise_data_dir = os.path.join(data_dir, "noise")  # 存储带噪声数据集的路径

    model_dir = "./ckpt/cifar-10"  # CIFAR-10 模型保存的路径
    noise_model_dir = os.path.join(model_dir, "noise")  # 存储带噪声模型的路径
    normal_model_dir = os.path.join(model_dir, "normal")  # 存储正常模型的路径

    # Step 1: 创建 D_0, D_inc 和 测试集数据
    train_data_Dinc, train_labels_Dinc, test_data, test_labels = (
        create_cifar10_npy_files(data_dir, noise_data_dir)
    )

    # Step 2: 校验生成的npy文件
    validate_npy_files(noise_data_dir)

    # Step 3: 构建增量数据集 D_tr 并保存
    forget_classes = torch.tensor([1, 3, 5, 7, 9])  # 假设选择这5个类作为遗忘类
    D_tr_data, D_tr_labels = create_incremental_data(
        train_data_Dinc, train_labels_Dinc, forget_classes, save_dir=noise_data_dir
    )

    # 校验 D_tr 数据集是否正确
    validate_d_tr(noise_data_dir)

    # 使用原始的完整CIFAR-10数据集进行训练（无增量、无噪声）
    original_train_data, original_train_labels = train_data_Dinc, train_labels_Dinc

    # Step 4: 定义并训练两个模型：一个基于增量数据集，一个基于原始CIFAR-10数据集
    incremental_model = IncrementalLearningModel(num_classes=10)
    original_model = IncrementalLearningModel(num_classes=10)

    # 确保存储模型的目录存在
    os.makedirs(noise_model_dir, exist_ok=True)
    os.makedirs(normal_model_dir, exist_ok=True)

    # 设置模型保存路径
    incremental_model_path = os.path.join(
        noise_model_dir, "incremental_model-cifar10.pth"
    )
    original_model_path = os.path.join(normal_model_dir, "original_model-cifar10.pth")

    # 训练两个模型并保存
    trained_incremental_model, trained_original_model = train_models(
        D_tr_data,
        D_tr_labels,
        original_train_data,
        original_train_labels,
        test_data,
        test_labels,
        incremental_model,
        original_model,
        device=device,
        incremental_save_path=incremental_model_path,
        original_save_path=original_model_path,
    )

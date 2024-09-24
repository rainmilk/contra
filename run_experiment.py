import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


def train_model(model, data, labels, epochs=10, batch_size=32, learning_rate=0.001):
    """
    训练模型函数
    :param model: 要训练的 ResNet 模型
    :param data: 输入的数据集
    :param labels: 输入的数据标签
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :return: 训练后的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(data.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


def load_model(model_path):
    """
    加载训练好的模型
    :param model_path: 模型文件路径
    :return: 加载的模型
    """
    model = models.resnet18(num_classes=10)  # 假设为 CIFAR-10 的 10 个类别
    model.load_state_dict(torch.load(model_path))
    return model


def train_step(step, subdir, output_dir="models"):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param output_dir: 模型保存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    if step == 0:
        # Step 1: 训练 M_p0
        D_0_data = torch.load(os.path.join(subdir, "D_0.npy"))
        D_0_labels = torch.load(os.path.join(subdir, "D_0_labels.npy"))

        model_p0 = models.resnet18(num_classes=10)
        print("开始训练 M_p0...")
        model_p0 = train_model(model_p0, D_0_data, D_0_labels, epochs=10)
        model_p0_path = os.path.join(output_dir, "model_p0.pth")
        torch.save(model_p0.state_dict(), model_p0_path)
        print(f"M_p0 训练完毕并保存至 {model_p0_path}")

    elif step == 1:
        # Step 2: 训练 M_p1
        model_p0_path = os.path.join(output_dir, "model_p0.pth")
        if not os.path.exists(model_p0_path):
            raise FileNotFoundError(f"模型文件 {model_p0_path} 未找到。请先训练 M_p0 (step=0)。")

        model_p0_loaded = load_model(model_p0_path)

        D_tr_1_data = torch.tensor(np.load(os.path.join(subdir, "cifar10_D_tr_data_version_1.npy")))
        D_tr_1_labels = torch.tensor(np.load(os.path.join(subdir, "cifar10_D_tr_labels_version_1.npy")))

        model_p1 = models.resnet18(num_classes=10)
        model_p1.load_state_dict(model_p0_loaded.state_dict())
        print("开始训练 M_p1...")
        model_p1 = train_model(model_p1, D_tr_1_data, D_tr_1_labels, epochs=10)
        model_p1_path = os.path.join(output_dir, "model_p1.pth")
        torch.save(model_p1.state_dict(), model_p1_path)
        print(f"M_p1 训练完毕并保存至 {model_p1_path}")

    elif step >= 2:
        # Step 3: 迭代训练 M_p2, M_p3, ..., M_p{step}
        for i in range(2, step + 1):
            prev_model_path = os.path.join(output_dir, f"model_p{i-1}.pth")
            if not os.path.exists(prev_model_path):
                raise FileNotFoundError(f"模型文件 {prev_model_path} 未找到。请先训练 M_p{i-1}。")

            model_prev = load_model(prev_model_path)

            D_tr_i_data = torch.tensor(np.load(os.path.join(subdir, f"cifar10_D_tr_data_version_{i}.npy")))
            D_tr_i_labels = torch.tensor(np.load(os.path.join(subdir, f"cifar10_D_tr_labels_version_{i}.npy")))

            model_current = models.resnet18(num_classes=10)
            model_current.load_state_dict(model_prev.state_dict())
            print(f"开始训练 M_p{i}...")
            model_current = train_model(model_current, D_tr_i_data, D_tr_i_labels, epochs=10)
            model_current_path = os.path.join(output_dir, f"model_p{i}.pth")
            torch.save(model_current.state_dict(), model_current_path)
            print(f"M_p{i} 训练完毕并保存至 {model_current_path}")

    else:
        raise ValueError("无效的步骤参数。请选择 step >= 0。")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet models step by step.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Specify the step to execute: 0 for M_p0, 1 for M_p1, or 2+ for M_p2, M_p3, etc.",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.2,
        help="噪声比例，与生成数据时使用的参数相匹配。",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="标签噪声类型，与生成数据时使用的参数相匹配。",
    )
    parser.add_argument(
        "--noise_dir",
        type=str,
        default="./data/cifar-10/noise",
        help="噪声数据集的根目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="训练好的模型的保存目录。",
    )

    args = parser.parse_args()

    # 构建子目录路径
    subdir = os.path.join(
        args.noise_dir,
        f"nr_{args.noise_ratio}_nt_{args.noise_type}"
    )

    if not os.path.exists(subdir):
        raise FileNotFoundError(f"数据子目录 {subdir} 不存在。请确保已生成相应的数据集。")

    print(f"使用数据子目录: {subdir}")
    train_step(args.step, subdir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

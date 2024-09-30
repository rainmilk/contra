import os
import warnings
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


def train_model(
    model,
    data,
    labels,
    test_data,
    test_labels,
    epochs=50,
    batch_size=32,
    optimizer_type="adam",
    learning_rate=0.01,
):
    """
    训练模型函数
    :param model: 要训练的 ResNet 模型
    :param data: 输入的数据集
    :param labels: 输入的数据标签
    :param test_data: 测试集数据
    :param test_labels: 测试集标签
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    :return: 训练后的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 根据用户选择的优化器初始化
    if optimizer_type == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    elif optimizer_type == "sgd": # add weight_decay, 0.7/0.8
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    dataset = torch.utils.data.TensorDataset(data.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(
        test_data.to(device), test_labels.to(device)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    iters = len(test_loader)
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/iters)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_outputs = model(test_inputs)
                _, predicted = torch.max(test_outputs, 1)
                total += test_targets.size(0)
                correct += (predicted == test_targets).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")

    return model


def load_model(model_path, num_classes):
    """
    加载训练好的模型
    :param model_path: 模型文件路径
    :param num_classes: 分类类别数
    :return: 加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到。")

    model = models.resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def train_step(
    step,
    subdir,
    ckpt_subdir,
    output_dir="ckpt",
    dataset_type="cifar-10",
    load_model_path=None,
    epochs=50,
    batch_size=32,
    optimizer_type="adam",
    learning_rate=0.001,
):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param ckpt_subdir: 模型检查点子目录路径
    :param output_dir: 模型保存目录
    :param dataset_type: 使用的数据集类型（cifar-10 或 cifar-100）
    :param load_model_path: 指定加载的模型路径（可选）
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    """
    warnings.filterwarnings("ignore")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_subdir, exist_ok=True)

    num_classes = 10 if dataset_type == "cifar-10" else 100

    # 加载训练和测试数据集
    D_test_data = torch.load(os.path.join(subdir, "test_data.npy"))
    D_test_labels = torch.load(os.path.join(subdir, "test_labels.npy"))

    # 打印当前执行的参数
    print(f"===== 执行步骤: {step} =====")
    print(f"数据子目录: {subdir}")
    print(f"检查点目录: {ckpt_subdir}")
    print(f"输出目录: {output_dir}")
    print(f"数据集类型: {dataset_type}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    if step == 0:  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        model_p0 = models.resnet18(num_classes=num_classes)
        print(f"开始训练 M_p0 ({dataset_type})...")

        D_train_data = torch.load(os.path.join(subdir, f"D_0.npy"))
        D_train_labels = torch.load(os.path.join(subdir, f"D_0_labels.npy"))

        # 打印用于训练的模型和数据
        print("用于训练的数据: D_0.npy 和 D_0_labels.npy")
        print("用于训练的模型: ResNet18 初始化")

        model_p0 = train_model(
            model_p0,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        model_p0_path = os.path.join(ckpt_subdir, "model_p0.pth")
        torch.save(model_p0.state_dict(), model_p0_path)
        print(f"M_p0 训练完毕并保存至 {model_p0_path}")
    elif (
        step == 1
    ):  # load上一步训练好的M_p0模型，然后基于 D_tr_data_version_1 和 D_tr_labels_version_1 进行训练，得到M_p1
        if load_model_path:  # 只能是 M_p0 模型
            if "model_p0" not in load_model_path:
                raise ValueError("加载的模型必须是 M_p0 模型。")
            model_p0_loaded = load_model(load_model_path, num_classes)
            print(f"加载指定模型: {load_model_path}")
        else:  # 只能是 M_p0 模型
            model_p0_path = os.path.join(ckpt_subdir, "model_p0.pth")
            model_p0_loaded = load_model(model_p0_path, num_classes)
            print(f"加载模型: {model_p0_path}")

        D_train_data = torch.load(os.path.join(subdir, f"D_tr_data_version_{step}.npy"))
        D_train_labels = torch.load(
            os.path.join(subdir, f"D_tr_labels_version_{step}.npy")
        )

        # 打印用于训练的模型和数据
        print(
            f"用于训练的数据: D_tr_data_version_{step}.npy 和 D_tr_labels_version_{step}.npy"
        )
        print("用于训练的模型: M_p0")

        model_p1 = models.resnet18(num_classes=num_classes)
        model_p1.load_state_dict(model_p0_loaded.state_dict())
        print(f"开始训练 M_p1 ({dataset_type})...")
        model_p1 = train_model(
            model_p1,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
        )
        model_p1_path = os.path.join(ckpt_subdir, "model_p1.pth")
        torch.save(model_p1.state_dict(), model_p1_path)
        print(f"M_p1 训练完毕并保存至 {model_p1_path}")
    else:  # 从外部加载通过命令行指定的某个模型
        if load_model_path:
            model_prev = load_model(load_model_path, num_classes)
            print(f"加载指定模型: {load_model_path}")
        else:  # 如果 step>=2 且用户通过命令行没有提供外部模型，则加载前一个模型
            prev_model_path = os.path.join(ckpt_subdir, f"model_p{step-1}.pth")
            if not os.path.exists(prev_model_path):
                raise FileNotFoundError(
                    f"模型文件 {prev_model_path} 未找到。请先训练 M_p{step-1}。"
                )
            model_prev = load_model(prev_model_path, num_classes)
            print(f"加载模型: {prev_model_path}")

        # 加载当前步骤的训练数据
        D_train_data = torch.load(os.path.join(subdir, f"D_tr_data_version_{step}.npy"))
        D_train_labels = torch.load(
            os.path.join(subdir, f"D_tr_labels_version_{step}.npy")
        )

        # 打印用于训练的模型和数据
        print(
            f"用于训练的数据: D_tr_data_version_{step}.npy 和 D_tr_labels_version_{step}.npy"
        )
        print(f"用于训练的模型: M_p{step-1}")

        # 训练当前模型
        model_current = models.resnet18(num_classes=num_classes)
        model_current.load_state_dict(model_prev.state_dict())
        print(f"开始训练 M_p{step} ({dataset_type})...")

        model_current = train_model(
            model_current,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
        )
        model_current_path = os.path.join(ckpt_subdir, f"model_p{step}.pth")
        torch.save(model_current.state_dict(), model_current_path)
        print(f"M_p{step} 训练完毕并保存至 {model_current_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet models step by step.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Specify the step to execute: 0 for M_p0, 1 for M_p1, etc.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["cifar-10", "cifar-100", "food-101"],
        required=True,
        help="选择数据集类型 (cifar-10 或 cifar-100)",
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
        "--gen_dir",
        type=str,
        default="./data",
        help="生成数据集的根目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpt",
        help="训练好的模型的保存目录。",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="指定要加载的模型文件路径（可选）。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练的轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每批训练样本数",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="选择优化器 (adam 或 sgd)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="学习率",
    )

    args = parser.parse_args()

    # 构建数据子目录路径
    subdir = os.path.join(
        args.gen_dir,
        args.dataset_type,
        "gen",
        f"nr_{args.noise_ratio}_nt_{args.noise_type}",
    )

    if not os.path.exists(subdir):
        raise FileNotFoundError(
            f"数据子目录 {subdir} 不存在。请确保已生成相应的数据集。"
        )

    # 构建模型检查点子目录路径
    ckpt_subdir = os.path.join(
        args.output_dir,
        args.dataset_type,
        f"nr_{args.noise_ratio}_nt_{args.noise_type}",
    )

    print(f"使用数据子目录: {subdir}")
    print(f"模型将保存至: {ckpt_subdir}")

    train_step(
        args.step,
        subdir,
        ckpt_subdir,
        output_dir=ckpt_subdir,
        dataset_type=args.dataset_type,
        load_model_path=args.load_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

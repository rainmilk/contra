import os
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from core_model.optimizer import create_optimizer_scheduler

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


# Warmup学习率调度器
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (self.last_epoch + 1) / self.total_iters
            for base_lr in self.base_lrs
        ]


def load_dataset(subdir, dataset_name, file_name, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    file_path = os.path.join(subdir, file_name)
    data = np.load(file_path)

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


def train_model(
    model,
    data,
    labels,
    test_data,
    test_labels,
    epochs=50,
    batch_size=32,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-4,
    writer=None,
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

    # 根据用户选择的优化器初始化
    optimizer, scheduler = create_optimizer_scheduler(optimizer_type, learning_rate, weight_decay,
                                                      step_size=epochs//10, gamma=0.5)

    dataset = torch.utils.data.TensorDataset(data.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(
        test_data.to(device), test_labels.to(device)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 用于存储训练和测试的损失和准确率
    train_losses = []
    test_accuracies = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm 进度条显示
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        # 更新学习率调度器
        scheduler.step()

        # 打印训练集的平均损失和准确率
        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
        )

        # TensorBoard记录
        if writer:
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Train/Accuracy", accuracy * 100, epoch)

        # 测试集评估
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(
                    device
                )
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += test_targets.size(0)
                correct_test += (predicted_test == test_targets).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%")

        if writer:
            writer.add_scalar("Test/Loss", test_loss, epoch)
            writer.add_scalar("Test/Accuracy", test_accuracy, epoch)

        model.train()

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
    dataset_name="cifar-10",
    load_model_path=None,
    epochs=50,
    batch_size=32,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-4,
    writer=None,
):
    """
    根据步骤训练模型
    :param step: 要执行的步骤（0, 1, 2, ...）
    :param subdir: 数据子目录路径
    :param ckpt_subdir: 模型检查点子目录路径
    :param output_dir: 模型保存目录
    :param dataset_name: 使用的数据集类型（cifar-10 或 cifar-100）
    :param load_model_path: 指定加载的模型路径（可选）
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :optimizer_type: 优化器
    :param learning_rate: 学习率
    """
    warnings.filterwarnings("ignore")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_subdir, exist_ok=True)

    # num_classes = 10 if dataset_name == "cifar-10" else 100

    # 根据 dataset_name 设置分类类别数
    if dataset_name == "cifar-10":
        num_classes = 10
    elif dataset_name == "cifar-100":
        num_classes = 100
    elif dataset_name == "food-101":
        num_classes = 101
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    # 加载训练和测试数据集
    D_test_data = load_dataset(subdir, dataset_name, "test_data.npy", is_data=True)
    D_test_labels = load_dataset(subdir, dataset_name, "test_labels.npy", is_data=False)

    # 打印当前执行的参数
    print(f"===== 执行步骤: {step} =====")
    print(f"数据子目录: {subdir}")
    print(f"检查点目录: {ckpt_subdir}")
    print(f"输出目录: {output_dir}")
    print(f"数据集类型: {dataset_name}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    if step == (-1):

        D_train_data = load_dataset(
            subdir, dataset_name, f"train_data.npy", is_data=True
        )
        D_train_labels = load_dataset(
            subdir, dataset_name, f"train_labels.npy", is_data=False
        )

        # 打印用于训练的模型和数据
        print("用于训练的数据: train_data.npy 和 train_labels.npy")
        print("用于训练的模型: ResNet18 初始化")

        model_raw = models.resnet18(num_classes=num_classes)
        print(f"开始训练 M_p0 ({dataset_name})...")

        model_raw = train_model(
            model_raw,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            writer=writer,
        )
        model_raw_path = os.path.join(ckpt_subdir, "model_raw.pth")
        torch.save(model_raw.state_dict(), model_raw_path)
        print(f"M_p0 训练完毕并保存至 {model_raw_path}")
        return

    if step == 0:  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        D_train_data = load_dataset(subdir, dataset_name, f"D_0.npy", is_data=True)
        D_train_labels = load_dataset(
            subdir, dataset_name, f"D_0_labels.npy", is_data=False
        )

        # 打印用于训练的模型和数据
        print("用于训练的数据: D_0.npy 和 D_0_labels.npy")
        print("用于训练的模型: ResNet18 初始化")

        model_p0 = models.resnet18(num_classes=num_classes)
        print(f"开始训练 M_p0 ({dataset_name})...")

        model_p0 = train_model(
            model_p0,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            writer=writer,
        )
        model_p0_path = os.path.join(ckpt_subdir, "model_p0.pth")
        torch.save(model_p0.state_dict(), model_p0_path)
        print(f"M_p0 训练完毕并保存至 {model_p0_path}")

    elif (
        step == 1
    ):  # load上一步训练好的M_p0模型，然后基于 D_tr_data_version_1 和 D_tr_labels_version_1 进行训练，得到M_p1

        D_train_data = load_dataset(
            subdir, dataset_name, f"D_tr_data_version_{step}.npy", is_data=True
        )
        D_train_labels = load_dataset(
            subdir, dataset_name, f"D_tr_labels_version_{step}.npy", is_data=False
        )

        if load_model_path:  # 只能是 M_p0 模型
            if "model_p0" not in load_model_path:
                raise ValueError("加载的模型必须是 M_p0 模型。")
            model_p0_loaded = load_model(load_model_path, num_classes)
            print(f"加载指定模型: {load_model_path}")
        else:  # 只能是 M_p0 模型
            model_p0_path = os.path.join(ckpt_subdir, "model_p0.pth")
            model_p0_loaded = load_model(model_p0_path, num_classes)
            print(f"加载模型: {model_p0_path}")

        # 打印用于训练的模型和数据
        print(
            f"用于训练的数据: D_tr_data_version_{step}.npy 和 D_tr_labels_version_{step}.npy"
        )
        print("用于训练的模型: M_p0")

        model_p1 = models.resnet18(num_classes=num_classes)
        model_p1.load_state_dict(model_p0_loaded.state_dict())
        print(f"开始训练 M_p1 ({dataset_name})...")
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
            weight_decay=weight_decay,
            writer=writer,
        )
        model_p1_path = os.path.join(ckpt_subdir, "model_p1.pth")
        torch.save(model_p1.state_dict(), model_p1_path)
        print(f"M_p1 训练完毕并保存至 {model_p1_path}")

    elif step >= 2:  # 从外部加载通过命令行指定的某个模型
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
        D_train_data = load_dataset(
            subdir, dataset_name, f"D_tr_data_version_{step}.npy", is_data=True
        )
        D_train_labels = load_dataset(
            subdir, dataset_name, f"D_tr_labels_version_{step}.npy", is_data=False
        )

        # 打印用于训练的模型和数据
        print(
            f"用于训练的数据: D_tr_data_version_{step}.npy 和 D_tr_labels_version_{step}.npy"
        )
        print(f"用于训练的模型: M_p{step-1}")

        # 训练当前模型
        model_current = models.resnet18(num_classes=num_classes)
        model_current.load_state_dict(model_prev.state_dict())
        print(f"开始训练 M_p{step} ({dataset_name})...")

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
            weight_decay=weight_decay,
            writer=writer,
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
        "--dataset_name",
        type=str,
        choices=["cifar-10", "cifar-100", "food-101"],
        required=True,
        help="选择数据集类型 (cifar-10 或 cifar-100 或 food-101)",
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
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减系数",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="是否使用类均衡的数据划分方式。如果不指定，则使用随机划分。",
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard for logging."
    )

    args = parser.parse_args()

    if args.balanced == True:
        # 构建数据子目录路径
        subdir = os.path.join(
            args.gen_dir,
            args.dataset_name,
            "gen",
            f"nr_{args.noise_ratio}_nt_{args.noise_type}_balanced",
        )
    else:
        # 构建数据子目录路径
        subdir = os.path.join(
            args.gen_dir,
            args.dataset_name,
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
        args.dataset_name,
        f"nr_{args.noise_ratio}_nt_{args.noise_type}",
    )

    print(f"使用数据子目录: {subdir}")
    print(f"模型将保存至: {ckpt_subdir}")

    writer = SummaryWriter(log_dir="runs/experiment") if args.use_tensorboard else None

    train_step(
        args.step,
        subdir,
        ckpt_subdir,
        output_dir=ckpt_subdir,
        dataset_name=args.dataset_name,
        load_model_path=args.load_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

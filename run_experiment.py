import os
import warnings
import numpy as np
import argparse

import torchvision.models
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from core_model.optimizer import create_optimizer_scheduler
from core_model.custom_model import ClassifierWrapper, load_custom_model

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class BaseTensorDataset(Dataset):

    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms is not None:
            self.transforms(data)

        return data, self.labels[index]


def get_num_of_classes(dataset_name):
    # 根据 dataset_name 设置分类类别数
    if dataset_name == "cifar-10":
        num_classes = 10
    elif dataset_name == "pet-37":
        num_classes = 37
    elif dataset_name == "cifar-100":
        num_classes = 100
    elif dataset_name == "food-101":
        num_classes = 101
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return num_classes


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
    batch_size=256,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=5e-4,
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

    optimizer, scheduler = create_optimizer_scheduler(
        optimizer_type=optimizer_type,
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
    )

    # weights = torchvision.models.ResNet18_Weights.DEFAULT
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15)
        ]
    )
    transform_test = transforms.Compose(
        [
            # weights.transforms()
        ]
    )

    dataset = BaseTensorDataset(data.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = BaseTensorDataset(test_data.to(device), test_labels.to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 用于存储训练和测试的损失和准确率
    train_losses = []
    test_accuracies = []

    num_classes = len(set(labels.tolist()))
    from torchvision.transforms import v2

    cutmix_transform = v2.CutMix(alpha=1.0, num_classes=num_classes)
    mixup_transform = v2.MixUp(alpha=0.5, num_classes=num_classes)
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        running_loss = 0.0
        correct = 0
        total = 0

        # 更新学习率调度器
        scheduler.step(epoch)

        # tqdm 进度条显示
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1} Training") as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                if np.random.rand() < 0.5:
                    inputs, targets = mixup_transform(inputs, targets)
                else:
                    inputs, targets = cutmix_transform(inputs, targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, mixed_max = torch.max(targets.data, 1)
                total += targets.size(0)
                correct += (predicted == mixed_max).sum().item()

                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

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
            with tqdm(
                total=len(test_loader), desc=f"Epoch {epoch + 1} Testing"
            ) as pbar:
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

                    # 更新进度条
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)

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

    # model = load_model()
    model = ClassifierWrapper(
        load_custom_model(model_name, num_classes), num_classes=num_classes
    )
    # model.load_state_dict(torch.load(model_path))
    return model


def train_step(
    step,
    subdir,
    ckpt_subdir,
    output_dir="ckpt",
    dataset_name="cifar-10",
    load_model_path=None,
    epochs=50,
    batch_size=256,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_decay=1e-4,
    writer=None,
    args=None,
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
    num_classes = get_num_of_classes(dataset_name)

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

    model_name = args.model
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

        model_raw = load_custom_model(model_name, num_classes)
        model_raw = ClassifierWrapper(
            model_raw, num_classes=num_classes, spectral_norm=False
        )
        print(f"开始训练 M_raw on ({dataset_name})...")

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
        print(f"M_raw 训练完毕并保存至 {model_raw_path}")
        return

    if step == 0:  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        D_train_data = load_dataset(subdir, dataset_name, f"D_0.npy", is_data=True)
        D_train_labels = load_dataset(
            subdir, dataset_name, f"D_0_labels.npy", is_data=False
        )

        # 打印用于训练的模型和数据
        print("用于训练的数据: D_0.npy 和 D_0_labels.npy")
        print("用于训练的模型: ResNet18 初始化")

        model_p0 = load_custom_model(model_name, num_classes)
        model_p0 = ClassifierWrapper(model_p0, num_classes)
        print(f"开始训练 M_p0 on ({dataset_name})...")

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

        # 打印用于训练的模型和数据
        print(
            f"用于训练的数据: D_tr_data_version_{step}.npy 和 D_tr_labels_version_{step}.npy"
        )
        print("用于训练的模型: M_p0")

        if load_model_path:  # 只能是 M_p0 模型
            if "model_p0" not in load_model_path:
                raise ValueError("加载的模型必须是 M_p0 模型。")
                return

        model_p0_path = os.path.join(ckpt_subdir, "model_p0.pth")
        print(f"加载模型: {model_p0_path}")

        # load model p0
        model_p0_loaded = load_custom_model(
            model_name=model_name, num_classes=num_classes, ckpt_path=model_p0_path
        )
        model_p0_loaded = ClassifierWrapper(model_p0_loaded, num_classes)

        # prepare model p1
        model_p1 = ClassifierWrapper(
            load_custom_model(model_name, num_classes), num_classes=num_classes
        )
        model_p1.load_state_dict(model_p0_loaded.state_dict())

        print(f"开始训练 M_p1 on ({dataset_name})...")
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

        # save model p1
        model_p1_path = os.path.join(ckpt_subdir, "model_p1.pth")
        torch.save(model_p1.state_dict(), model_p1_path)
        print(f"M_p1 训练完毕并保存至 {model_p1_path}")

    elif step >= 2:  # 从外部加载通过命令行指定的某个模型
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

        if load_model_path:  # 只能是 M_p0 模型
            if "model_p" not in load_model_path:
                raise ValueError("加载的模型名称必须是 M_px(x表示序号1,2...)。")
                return

        prev_model_path = os.path.join(ckpt_subdir, f"model_p{step-1}.pth")
        print(f"加载模型: {prev_model_path}")

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"模型文件 {prev_model_path} 未找到。请先训练 M_p{step-1}。"
            )

        prev_model_loaded = load_custom_model(
            model_name=model_name, num_classes=num_classes, ckpt_path=prev_model_path
        )
        prev_model_loaded = ClassifierWrapper(prev_model_loaded, num_classes)

        current_model = ClassifierWrapper(
            load_custom_model(model_name, num_classes), num_classes=num_classes
        )
        current_model.load_state_dict(prev_model_loaded.state_dict())
        print(f"开始训练 M_p{step} on ({dataset_name})...")

        current_model = train_model(
            current_model,
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

        # save current model
        current_model_path = os.path.join(ckpt_subdir, f"model_p{step}.pth")
        torch.save(current_model.state_dict(), current_model_path)
        print(f"M_p{step} 训练完毕并保存至 {current_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train models step by step.")

    parser.add_argument(
        "--model",
        type=str,
        choices=["cifar-resnet18", "cifar-wideresnet40", "resnet18", "vgg19"],
        required=True,
        help="Select in (cifar-resnet18, cifar-wideresnet40, resnet18, vgg19)",
    )

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
        "--load_model_path",
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
        default=256,
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
        default=5e-4,
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
        load_model_path=args.load_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        writer=writer,
        args=args,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

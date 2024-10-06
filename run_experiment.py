import os
import warnings
import numpy as np
from args_paser import parse_args

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from core_model.optimizer import create_optimizer_scheduler
from core_model.custom_model import ClassifierWrapper, load_custom_model
from configs import settings

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import v2


class BaseTensorDataset(Dataset):

    def __init__(self, data, labels, transforms=None, device=None):
        self.data = torch.as_tensor(data, device=device)
        self.labels = torch.as_tensor(labels, device=device)
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

    dataset = BaseTensorDataset(data, labels, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    test_dataset = BaseTensorDataset(test_data, test_labels, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 用于存储训练和测试的损失和准确率
    train_losses = []
    test_accuracies = []

    num_classes = len(set(labels.tolist()))

    use_data_aug = True

    if use_data_aug:
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
                if use_data_aug:
                    transform = np.random.choice([mixup_transform, cutmix_transform])
                    inputs, targets = transform(inputs, targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                mixed_max = torch.argmax(targets.data, 1) if use_data_aug else targets
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


def train_step(
    args,
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

    # num_classes = 10 if dataset_name == "cifar-10" else 100
    dataset_name = args.dataset
    num_classes = get_num_of_classes(dataset_name)

    # 打印当前执行的参数
    print(f"===== 执行步骤: {args.step} =====")
    print(f"数据集类型: {dataset_name}")
    print(f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}")

    model_name = args.model
    step = args.step
    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    uni_name = args.uni_name
    if step < 0:

        D_train_data = np.load(settings.get_dataset_path(dataset_name, case, "train_data"))
        D_train_labels = np.load(settings.get_dataset_path(dataset_name, case, "train_label"))
        D_test_data = np.load(settings.get_dataset_path(dataset_name, case, "test_data"))
        D_test_labels = np.load(settings.get_dataset_path(dataset_name, case, "test_label"))

        # 打印用于训练的模型和数据
        print("用于训练的数据: train_data.npy 和 train_labels.npy")
        print("用于训练的模型: ResNet18 初始化")

        model_raw = load_custom_model(model_name, num_classes)
        model_raw = ClassifierWrapper(
            model_raw, num_classes=num_classes, freeze_weights=False
        )
        print(f"开始训练 M_raw on ({dataset_name})...")

        model_raw = train_model(
            model_raw,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            writer=writer,
        )
        model_raw_path = settings.get_ckpt_path(dataset_name, case, model_name,
                                                "worker_restore", unique_name=uni_name)
        subdir = os.path.dirname(model_raw_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_raw.state_dict(), model_raw_path)
        print(f"M_raw 训练完毕并保存至 {model_raw_path}")
        return
    elif step == 0:  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        D_train_data = np.load(settings.get_dataset_path(dataset_name, case, "train_data", step=step))
        D_train_labels = np.load(settings.get_dataset_path(dataset_name, case, "train_label", step=step))
        D_test_data = np.load(settings.get_dataset_path(dataset_name, case, "test_data"))
        D_test_labels = np.load(settings.get_dataset_path(dataset_name, case, "test_label"))

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
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            writer=writer,
        )
        model_p0_path = settings.get_ckpt_path(dataset_name, case,
                                               model_name, "worker_restore",
                                               step=step, unique_name=uni_name)
        subdir = os.path.dirname(model_p0_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_p0.state_dict(), model_p0_path)
        print(f"M_p0 训练完毕并保存至 {model_p0_path}")
    else:  # 从外部加载通过命令行指定的某个模型
        # 加载当前步骤的训练数据
        D_train_data = np.load(settings.get_dataset_path(dataset_name, case, "train_data", step=step))
        D_train_labels = np.load(settings.get_dataset_path(dataset_name, case, "train_label", step=step))
        D_test_data = np.load(settings.get_dataset_path(dataset_name, case, "test_data"))
        D_test_labels = np.load(settings.get_dataset_path(dataset_name, case, "test_label"))

        # 打印用于训练的模型和数据
        print(f"用于训练的模型: M_p{step-1}")

        prev_model_path = settings.get_ckpt_path(dataset_name, case, model_name,
                                                 "worker_restore",
                                                 step=step-1, unique_name=uni_name)
        print(f"加载模型: {prev_model_path}")

        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"模型文件 {prev_model_path} 未找到。请先训练 M_p{step-1}。"
            )

        model_loaded = load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        current_model = ClassifierWrapper(model_loaded, num_classes)
        current_model.load_state_dict(torch.load(prev_model_path))

        print(f"开始训练 M_p{step} on ({dataset_name})...")

        current_model = train_model(
            current_model,
            D_train_data,
            D_train_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            writer=writer,
        )

        # save current model
        current_model_path = settings.get_ckpt_path(dataset_name, case,
                                               model_name, "working_restore",
                                               step=step, unique_name=uni_name)
        subdir = os.path.dirname(current_model_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(current_model.state_dict(), current_model_path)
        print(f"M_p{step} 训练完毕并保存至 {current_model_path}")


def main():
    args = parse_args()

    writer = SummaryWriter(log_dir="runs/experiment") if args.use_tensorboard else None

    train_step(
        args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

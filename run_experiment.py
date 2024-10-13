import os
import shutil
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
from train_test_utils import train_model


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


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    data = np.load(file_path)

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor


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
    print(
        f"Epochs: {args.num_epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}"
    )

    model_name = args.model
    step = args.step
    case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
    uni_name = args.uni_name
    model_suffix = "worker_raw" if args.model_suffix is None else args.model_suffix
    if step < 0:

        D_train_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "train_data")
        )
        D_train_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "train_label")
        )
        D_test_data = np.load(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "test_label")
        )

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
            num_classes,
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
        model_raw_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "worker_restore", unique_name=uni_name
        )
        subdir = os.path.dirname(model_raw_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_raw.state_dict(), model_raw_path)
        print(f"M_raw 训练完毕并保存至 {model_raw_path}")
        return
    elif step == 0:  # 基于$D_0$数据集和原始的resnet网络训练一个模型 M_p0
        # model_p0_path = settings.get_pretrain_ckpt_path(
        #     dataset_name, case, model_name, "worker_restore", step=step
        # )
        pretrain_case = "pretrain"
        model_p0_path = settings.get_ckpt_path(
            dataset_name, pretrain_case, model_name, "worker_restore", step=step
        )

        if uni_name is None:
            D_train_data = np.load(
                settings.get_dataset_path(dataset_name, case, "train_data", step=step)
            )
            D_train_labels = np.load(
                settings.get_dataset_path(dataset_name, case, "train_label", step=step)
            )
            D_test_data = np.load(
                settings.get_dataset_path(dataset_name, case, "test_data")
            )
            D_test_labels = np.load(
                settings.get_dataset_path(dataset_name, case, "test_label")
            )

            # 打印用于训练的模型和数据
            print("用于训练的数据: D_0.npy 和 D_0_labels.npy")
            print("用于训练的模型: ResNet18 初始化")

            model_p0 = load_custom_model(model_name, num_classes)
            model_p0 = ClassifierWrapper(model_p0, num_classes)

            print(f"开始训练 M_p0 on ({dataset_name})...")

            model_p0 = train_model(
                model_p0,
                num_classes,
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
            subdir = os.path.dirname(model_p0_path)
            os.makedirs(subdir, exist_ok=True)
            torch.save(model_p0.state_dict(), model_p0_path)
            print(f"M_p0 训练完毕并保存至 {model_p0_path}")
        else:
            copy_model_p0_path = settings.get_ckpt_path(
                dataset_name,
                case,
                model_name,
                "worker_restore",
                step=step,
                unique_name=uni_name,
            )
            if os.path.exists(model_p0_path):
                subdir = os.path.dirname(copy_model_p0_path)
                os.makedirs(subdir, exist_ok=True)
                shutil.copy(model_p0_path, copy_model_p0_path)
                print(f"Copy {model_p0_path} to {copy_model_p0_path}")
            else:
                raise FileNotFoundError(model_p0_path)

    else:  # 从外部加载通过命令行指定的某个模型
        # 加载当前步骤的训练数据
        if args.train_aux:
            trainset = "aux"
            load_model_suffix = "worker_raw"
            data_step = None
            model_step = step
        else:
            trainset = "train"
            load_model_suffix = "worker_restore"
            data_step = step
            model_step = step - 1
        D_train_data = np.load(
            settings.get_dataset_path(
                dataset_name, case, f"{trainset}_data", step=data_step
            )
        )
        D_train_labels = np.load(
            settings.get_dataset_path(
                dataset_name, case, f"{trainset}_label", step=data_step
            )
        )
        D_test_data = np.load(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = np.load(
            settings.get_dataset_path(dataset_name, case, "test_label")
        )

        # 打印用于训练的模型和数据
        print(f"用于训练的模型: M_p{step-1}")

        prev_model_path = settings.get_ckpt_path(
            dataset_name,
            case,
            model_name,
            load_model_suffix,
            step=model_step,
            unique_name=uni_name,
        )
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
            num_classes,
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
        current_model_path = settings.get_ckpt_path(
            dataset_name,
            case,
            model_name,
            model_suffix,
            step=step,
            unique_name=uni_name,
        )
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

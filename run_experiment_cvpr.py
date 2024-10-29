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

conference_name = "cvpr"

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
    elif dataset_name == "flower-102":
        num_classes = 102
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    return num_classes


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101, pet-37, flower-102)
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
    
    if conference_name is not None:
        case = f"nr_{args.noise_ratio}_nt_{args.noise_type}_{conference_name}"
    
    uni_name = args.uni_name
    model_suffix = "worker_raw" if args.model_suffix is None else args.model_suffix

    if step == 0:  # Step 0: Train M_0 on D_0 (clean dataset)
        D_0_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_0_data")
        )
        D_0_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_0_labels"), is_data=False
        )
        D_test_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_label"), is_data=False
        )

        # 打印用于训练的模型和数据
        print("用于训练的数据: D_0_data 和 D_0_labels")
        print("用于训练的模型: ResNet18 初始化")

        model_0 = load_custom_model(model_name, num_classes)
        model_0 = ClassifierWrapper(
            model_0, num_classes=num_classes, freeze_weights=False
        )
        print(f"开始训练 M_0 on ({dataset_name})...")

        model_0 = train_model(
            model_0,
            num_classes,
            D_0_data,
            D_0_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            writer=writer,
        )
        model_0_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "M_0", unique_name=uni_name
        )
        subdir = os.path.dirname(model_0_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_0.state_dict(), model_0_path)
        print(f"M_0 训练完毕并保存至 {model_0_path}")
        return

    elif step == 1:  # Step 1: Train M_1 on D_1+ (noisy dataset)
        D_1_plus_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_1_plus_data")
        )
        D_1_plus_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_1_plus_labels"),
            is_data=False,
        )
        D_test_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_label"), is_data=False
        )

        # 打印用于训练的模型和数据
        print("用于训练的数据: D_1_plus_data 和 D_1_plus_labels")
        print("用于训练的模型: 从 M_0 开始")

        prev_model_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "M_0", unique_name=uni_name
        )
        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"模型文件 {prev_model_path} 未找到。请先训练 M_0。"
            )

        model_1 = load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        model_1 = ClassifierWrapper(model_1, num_classes)
        model_1.load_state_dict(torch.load(prev_model_path))

        print(f"开始训练 M_1 on ({dataset_name})...")

        model_1 = train_model(
            model_1,
            num_classes,
            D_1_plus_data,
            D_1_plus_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            writer=writer,
        )

        model_1_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "M_1", unique_name=uni_name
        )
        subdir = os.path.dirname(model_1_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_1.state_dict(), model_1_path)
        print(f"M_1 训练完毕并保存至 {model_1_path}")
        return

    elif step == 2:  # Step 2: Repair M_1 using D_1-
        D_1_minus_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_1_minus_data")
        )
        D_1_minus_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "D_1_minus_labels"),
            is_data=False,
        )
        D_test_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_data")
        )
        D_test_labels = load_dataset(
            settings.get_dataset_path(dataset_name, case, "test_label"), is_data=False
        )

        # 使用基于 M_1 的修复方法
        prev_model_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "M_1", unique_name=uni_name
        )
        if not os.path.exists(prev_model_path):
            raise FileNotFoundError(
                f"模型文件 {prev_model_path} 未找到。请先训练 M_1。"
            )

        model_repaired = load_custom_model(
            model_name=model_name, num_classes=num_classes, load_pretrained=False
        )
        model_repaired = ClassifierWrapper(model_repaired, num_classes)
        model_repaired.load_state_dict(torch.load(prev_model_path))

        print(f"开始修复 M_1 on ({dataset_name}) 使用 D_1_minus...")

        model_repaired = train_model(
            model_repaired,
            num_classes,
            D_1_minus_data,
            D_1_minus_labels,
            D_test_data,
            D_test_labels,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_aug=args.data_aug,
            writer=writer,
        )

        model_repaired_path = settings.get_ckpt_path(
            dataset_name, case, model_name, "M_repaired", unique_name=uni_name
        )
        subdir = os.path.dirname(model_repaired_path)
        os.makedirs(subdir, exist_ok=True)
        torch.save(model_repaired.state_dict(), model_repaired_path)
        print(f"M_1 修复完毕并保存至 {model_repaired_path}")
        return


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
    
    '''
    # 以CIFAR-10为例，步骤如下：
    
    # 1. 使用 D_0 训练 M_0
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 0 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    # 2. 使用 D_1+ 训练 M_1
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 1 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    # 3. 使用 D_1- 训练（修复） M_1
    python run_experiment_cvpr.py --dataset cifar-10 --model resnet18 --data_dir data/cifar-10/gen --step 2 --num_epochs 50 --batch_size 64 --learning_rate 0.001 --gpu 7
    
    '''
    
    main()

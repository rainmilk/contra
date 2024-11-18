import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import random
import torch.nn as nn
import torchvision.models as models


def set_seed(seed):
    """
    设置随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_test_data(data_dir, batch_size=64):
    """
    加载测试数据集。

    :param data_dir: 存储 test_data.npy 和 test_labels.npy 的目录
    :param batch_size: 批次大小
    :return: DataLoader 对象
    """
    test_data_path = os.path.join(data_dir, "test_data.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")

    # 检查文件是否存在
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"{test_data_path} 不存在。请检查路径。")
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"{test_labels_path} 不存在。请检查路径。")

    # 使用 torch.load 加载数据
    test_data = torch.load(test_data_path)
    test_labels = torch.load(test_labels_path)

    # 转换为 torch.Tensor，如果数据没有问题可以跳过这步
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels)

    # 创建 TensorDataset 和 DataLoader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader


def load_model(model_path, device, num_classes=10):
    """
    加载训练好的模型。

    :param model_path: 模型文件路径
    :param device: 设备（CPU 或 GPU）
    :param num_classes: 类别数量
    :return: 加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。请检查路径。")

    # 实例化模型
    model = models.resnet18(num_classes=num_classes)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式

    return model


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型的准确率，并生成混淆矩阵和分类报告。

    :param model: 要评估的模型
    :param test_loader: 测试数据的 DataLoader
    :param device: 设备（CPU 或 GPU）
    :return: 准确率、混淆矩阵、分类报告
    """
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)

    return accuracy, conf_matrix, class_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and Test Incremental Learning Model on CIFAR-10"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model (.pth file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing test data (test_data.npy and test_labels.npy)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载测试数据
    print("Loading test data...")
    test_loader = load_test_data(args.data_dir, batch_size=args.batch_size)
    print(f"Test Dataset Size: {len(test_loader.dataset)}")

    # 加载模型
    print("Loading model...")
    model = load_model(args.model_path, device, num_classes=10)
    print("Model loaded successfully.")

    # 评估模型
    print("Evaluating model on test data...")
    test_accuracy, conf_matrix, class_report = evaluate_model(
        model, test_loader, device
    )

    # 打印评估结果
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # 可选：保存评估结果到文件
    evaluation_dir = os.path.dirname(args.model_path)
    accuracy_path = os.path.join(evaluation_dir, "test_accuracy.txt")
    conf_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.npy")
    class_report_path = os.path.join(evaluation_dir, "classification_report.txt")

    with open(accuracy_path, "w") as f:
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    np.save(conf_matrix_path, conf_matrix)

    with open(class_report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(class_report)

    print(f"\nEvaluation results saved to {evaluation_dir}")

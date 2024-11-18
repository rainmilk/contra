import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import json

import sys

# 将项目根目录加入到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset_animals_10 import get_animals_10_dataset
from utils.dataset_flowers_102 import get_flowers_102_dataset


# 定义性能指标保存函数
def save_performance_metrics(
    model_name, dataset_name, condition_name, accuracy, loss, save_path
):
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "condition": condition_name,
        "accuracy": accuracy,
        "loss": loss,
    }
    with open(save_path, "w") as f:
        json.dump(metrics, f)


# 1. 定义数据集加载函数
def get_test_loader(dataset_name, dataset_paths, batch_size=64):
    if dataset_name == "cifar-10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_paths[dataset_name],
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar-100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=dataset_paths[dataset_name],
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "animals-10":
        _, test_dataset = get_animals_10_dataset(dataset_paths[dataset_name])
    elif dataset_name == "flowers-102":
        _, test_dataset = get_flowers_102_dataset(dataset_paths[dataset_name])
    else:
        raise ValueError("Unsupported dataset")

    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )


# 2. 加载模型并测试
def load_and_test_model(pth_file, model_class, test_loader, device, num_classes):
    model = model_class(pretrained=False).to(device)

    # 修改模型分类器的最后一层，使其与训练时的类别数相匹配
    if model_class == torchvision.models.vgg16:
        model.classifier[6] = nn.Linear(4096, num_classes).to(device)  # 修改为与训练时类别数相同
    elif model_class == torchvision.models.resnet18:
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)  # ResNet 的分类器调整

    # 加载训练好的权重
    model.load_state_dict(torch.load(pth_file))
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Model"):
            images, labels = images.to(device), labels.to(device)  # 确保图像和标签都在同一设备
            outputs = model(images)  # 在 GPU 上运行推理
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)

    return accuracy, avg_loss


# 3. 遍历 models 目录中的所有 pth 文件并生成 json
def generate_json_for_all_pth():
    models_root = "./models"
    result_dir = "./result_analysis/results"
    os.makedirs(result_dir, exist_ok=True)

    dataset_paths = {
        "cifar-10": "./data/cifar-10",
        "cifar-100": "./data/cifar-100",
        "animals-10": "./data/animals-10",
        "flowers-102": "./data/flowers-102",
    }

    num_classes_dict = {
        "cifar-10": 10,
        "cifar-100": 100,
        "animals-10": 10,
        "flowers-102": 102,
    }

    # 遍历所有模型、数据集和条件目录
    for model_name in os.listdir(models_root):
        model_path = os.path.join(models_root, model_name)
        if os.path.isdir(model_path):
            for dataset_name in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_name)
                if os.path.isdir(dataset_path):
                    for condition_name in os.listdir(dataset_path):
                        condition_path = os.path.join(dataset_path, condition_name)
                        if os.path.isdir(condition_path):
                            for pth_file in os.listdir(condition_path):
                                if pth_file.endswith(".pth"):
                                    pth_file_path = os.path.join(
                                        condition_path, pth_file
                                    )
                                    print(f"Processing {pth_file_path}")

                                    # 选择模型类别
                                    if model_name == "resnet18":
                                        model_class = torchvision.models.resnet18
                                    elif model_name == "vgg16":
                                        model_class = torchvision.models.vgg16
                                    else:
                                        print(f"Unsupported model: {model_name}")
                                        continue

                                    # 加载测试集
                                    test_loader = get_test_loader(
                                        dataset_name, dataset_paths
                                    )

                                    # 使用 GPU 进行推理
                                    device = torch.device(
                                        "cuda" if torch.cuda.is_available() else "cpu"
                                    )

                                    # 获取数据集类别数量
                                    num_classes = num_classes_dict[dataset_name]

                                    # 加载并测试模型，获取性能
                                    accuracy, loss = load_and_test_model(
                                        pth_file_path,
                                        model_class,
                                        test_loader,
                                        device,
                                        num_classes,
                                    )

                                    # 保存性能指标为 JSON 文件
                                    save_path = os.path.join(
                                        result_dir,
                                        f"{model_name}_{dataset_name}_{condition_name}.json",
                                    )
                                    save_performance_metrics(
                                        model_name,
                                        dataset_name,
                                        condition_name,
                                        accuracy,
                                        loss,
                                        save_path,
                                    )
                                    print(f"Performance metrics saved to {save_path}")


if __name__ == "__main__":
    generate_json_for_all_pth()

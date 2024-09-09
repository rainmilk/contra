import torch
from torch.utils.data import DataLoader
from dataset_loader import DatasetLoader
from model_utils import ModelUtils
from train_test_utils import TrainTestUtils


def run_experiment(dataset_name, model_name, selected_classes, condition):
    # 数据集加载
    dataset_paths = {
        "cifar-10": "./data/cifar-10",
        "cifar-100": "./data/cifar-100",
        "animals-10": "./data/animals-10",
        "tiny-imagenet-200": "./data/tiny-imagenet-200",
    }

    num_classes_dict = {
        "cifar-10": 10,
        "cifar-100": 100,
        "animals-10": 10,
        "tiny-imagenet-200": 200,
    }

    dataset_loader = DatasetLoader(dataset_name, dataset_paths, num_classes_dict)
    train_dataset, test_dataset = dataset_loader.get_dataset()

    if condition == "original_data":
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=2
        )
    elif condition == "removed_50_percent":
        train_dataset_shifted = dataset_loader.remove_50_percent_of_selected_classes(
            train_dataset, selected_classes
        )
        train_loader = DataLoader(
            train_dataset_shifted, batch_size=64, shuffle=True, num_workers=2
        )
    elif condition == "noisy_data":
        train_dataset_shifted = dataset_loader.remove_50_percent_of_selected_classes(
            train_dataset, selected_classes
        )
        noisy_train_data = dataset_loader.add_noise_to_selected_classes(
            train_dataset_shifted, selected_classes
        )
        train_loader = DataLoader(
            noisy_train_data, batch_size=64, shuffle=True, num_workers=2
        )
    else:
        raise ValueError("Unsupported condition: " + condition)

    # 创建模型
    model_utils = ModelUtils(model_name, num_classes_dict[dataset_name])
    model = model_utils.create_resnet_model()

    # 损失函数和优化器
    criterion, optimizer = model_utils.get_criterion_and_optimizer(model)

    # 训练和测试工具
    train_test_utils = TrainTestUtils(model_name, dataset_name)
    save_path = train_test_utils.create_save_path(condition)
    train_test_utils.train_and_save(
        model, train_loader, criterion, optimizer, save_path
    )

    # 测试模型
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    train_test_utils.test(model, test_loader)


# 执行实验
if __name__ == "__main__":
    for dataset in ["cifar-10", "cifar-100"]:
        print(f"开始 {dataset} 的原始数据集训练")
        run_experiment(dataset, "resnet18", [0, 1, 2, 3, 4], "original_data")

        print(f"\n开始 {dataset} 的移除 50% 数据的训练")
        run_experiment(dataset, "resnet18", [0, 1, 2, 3, 4], "removed_50_percent")

        print(f"\n开始 {dataset} 的添加噪声数据的训练")
        run_experiment(dataset, "resnet18", [0, 1, 2, 3, 4], "noisy_data")

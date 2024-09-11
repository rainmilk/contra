import torch
from torch.utils.data import DataLoader
from dataset_loader import DatasetLoader
from model_utils import ModelUtils
from train_test_utils import TrainTestUtils
from tqdm import tqdm


def run_experiment(
    dataset_name,
    model_name,
    selected_classes_remove,
    selected_classes_noise,
    condition,
    remove_fraction,
    noise_type,
    noise_fraction,
    use_early_stopping,
):
    # 数据集加载
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

    # 显示数据集加载进度
    print(f"Loading dataset: {dataset_name}")
    dataset_loader = DatasetLoader(dataset_name, dataset_paths, num_classes_dict)

    # 添加数据加载进度条
    with tqdm(total=2, desc="Dataset Loading") as pbar:
        train_dataset, test_dataset = dataset_loader.get_dataset()
        pbar.update(1)

        # 处理不同实验条件的 DataLoader
        if condition == "original_data":
            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True, num_workers=2
            )
        elif condition == "remove_data":
            # 仅执行删除操作
            train_dataset_shifted = dataset_loader.remove_fraction_of_selected_classes(
                train_dataset, selected_classes_remove, remove_fraction=remove_fraction
            )
            train_loader = DataLoader(
                train_dataset_shifted, batch_size=64, shuffle=True, num_workers=2
            )
        elif condition == "noisy_data":
            # 仅执行噪声注入操作
            train_dataset_shifted = dataset_loader.add_noise_to_selected_classes(
                train_dataset,
                selected_classes_noise,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
            )
            train_loader = DataLoader(
                train_dataset_shifted, batch_size=64, shuffle=True, num_workers=2
            )
        elif condition == "all_perturbations":
            # 同时删除样本并注入噪声
            modified_dataset = dataset_loader.modify_dataset(
                dataset=train_dataset,
                selected_classes_remove=selected_classes_remove,
                selected_classes_noise=selected_classes_noise,
                remove_fraction=remove_fraction,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
            )
            train_loader = DataLoader(
                modified_dataset, batch_size=64, shuffle=True, num_workers=2
            )
        else:
            raise ValueError("Unsupported condition: " + condition)

        pbar.update(1)  # 数据集加载完成

    # 创建模型
    print(f"Creating model: {model_name}")
    model_utils = ModelUtils(model_name, num_classes_dict[dataset_name])
    model = model_utils.create_resnet_model()

    # 损失函数和优化器
    criterion, optimizer = model_utils.get_criterion_and_optimizer(model)

    # 训练和测试工具
    train_test_utils = TrainTestUtils(model_name, dataset_name)
    save_path = train_test_utils.create_save_path(condition)

    # Early stopping 条件
    early_stopping_patience = 10  # 连续 N 次性能下降则停止
    early_stopping_accuracy_threshold = 0.95  # 如果准确率达到 95%，提前停止
    best_accuracy = 0
    patience_counter = 0
    num_epochs = 200  # 最多训练 N 个epoch

    # 训练模型并显示进度
    print(f"Training model on {dataset_name} with condition: {condition}")
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        train_test_utils.train_and_save(
            model,
            train_loader,
            criterion,
            optimizer,
            save_path,
            epoch=epoch,
            num_epochs=num_epochs,
            save_final_model_only=True,
        )

        # 在验证集上评估模型性能
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=2
        )
        accuracy = train_test_utils.test(model, test_loader, condition)

        # 如果使用了early stopping
        if use_early_stopping:
            # 检查是否达到了早停的条件
            if accuracy >= early_stopping_accuracy_threshold:
                print(
                    f"Accuracy {accuracy*100:.2f}% reached threshold. Stopping early."
                )
                break

            # 检查性能是否下降
            if accuracy < best_accuracy:
                patience_counter += 1  # 如果性能下降，计数器加1
                print(
                    f"Performance dropped. Patience counter: {patience_counter}/{early_stopping_patience}"
                )
            elif accuracy > best_accuracy:
                # 如果准确率提高，更新best_accuracy并重置patience_counter
                best_accuracy = accuracy
                patience_counter = 0
            else:
                # 性能持平时也重置计数器（持平不算下降）
                patience_counter = 0

            # 如果连续下降次数达到早停条件，则提前停止训练
            if patience_counter >= early_stopping_patience:
                print(
                    f"Performance dropped for {early_stopping_patience} consecutive epochs. Stopping early."
                )
                break
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed without early stopping.")


# 执行实验
if __name__ == "__main__":
    for dataset in ["cifar-10", "cifar-100"]:
        print(f"开始 {dataset} 的原始数据集训练")
        run_experiment(
            dataset,
            "resnet18",
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            "original_data",
            remove_fraction=0.5,
            noise_type="gaussian",
            noise_fraction=0.1,
            use_early_stopping=False,
        )

        print(f"\n开始 {dataset} 的移除 50% 数据的训练")
        run_experiment(
            dataset,
            "resnet18",
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            "remove_data",
            remove_fraction=0.5,
            noise_type="gaussian",
            noise_fraction=0.1,
            use_early_stopping=True,
        )

        print(f"\n开始 {dataset} 的添加噪声数据的训练")
        run_experiment(
            dataset,
            "resnet18",
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            "noisy_data",
            remove_fraction=0.5,
            noise_type="gaussian",
            noise_fraction=0.1,
            use_early_stopping=True,
        )

        print(f"\n开始 {dataset} 的组合操作（删除样本+噪声注入）的训练")
        run_experiment(
            dataset,
            "resnet18",
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            "all_perturbations",
            remove_fraction=0.5,
            noise_type="salt_pepper",
            noise_fraction=0.2,
            use_early_stopping=False,
        )

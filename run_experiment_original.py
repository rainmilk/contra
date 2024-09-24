import torch
from torch.utils.data import DataLoader
from dataset_utils import DatasetUtils
from model_utils import ModelUtils
from train_test_utils import TrainTestUtils
from tqdm import tqdm


def run_experiment(
    dataset_name,
    model_name,
    condition,
    remove_fraction,
    noise_type,
    noise_fraction,
    use_early_stopping,
    selected_classes_remove=None,  # 设置默认值为 None
    selected_classes_noise=None,  # 设置默认值为 None
    batch_size=64,  # 默认batch_size
    learning_rate=0.001,  # 默认学习率
    optimizer="adam",
    momentum=0.9,  # 默认 momentum 值（用于SGD）
    weight_decay=1e-4,  # 默认 weight decay 值
    num_epochs=200,  # 默认epoch数量
    early_stopping_patience=10,  # 早停耐心值
    early_stopping_accuracy_threshold=0.95,  # 提前停止的准确率阈值
    pretrained=False,  # 增加pretrained参数
    **kwargs,  # 捕获额外的参数
):
    # 数据集路径和类别数量
    dataset_paths = {
        "cifar-10": "./data/cifar-10",
        "cifar-100": "./data/cifar-100",
        "flowers-102": "./data/flowers-102",
        "tiny-imagenet-200": "./data/tiny-imagenet-200",
    }

    num_classes_dict = {
        "cifar-10": 10,
        "cifar-100": 100,
        "flowers-102": 102,
        "tiny-imagenet-200": 200,
    }

    # 加载数据集
    print(f"Loading dataset: {dataset_name}")
    dataset_utils = DatasetUtils(dataset_name, dataset_paths, num_classes_dict)

    # 显示数据加载进度
    with tqdm(total=2, desc="Dataset Loading") as pbar:
        train_dataset, val_dataset, test_dataset = dataset_utils.get_dataset()
        pbar.update(1)

        # 处理不同实验条件的 DataLoader
        if condition == "original_data":
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )
        elif condition == "remove_data":
            if selected_classes_remove is None:
                raise ValueError("For 'remove_data', 'selected_classes_remove' must be provided.")
            # 仅执行删除操作
            train_dataset_shifted = dataset_utils.remove_fraction_of_selected_classes(
                train_dataset, selected_classes_remove, remove_fraction=remove_fraction
            )
            train_loader = DataLoader(
                train_dataset_shifted,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        elif condition == "noisy_data":
            if selected_classes_noise is None:
                raise ValueError("For 'noisy_data', 'selected_classes_noise' must be provided.")
            # 仅执行噪声注入操作
            train_dataset_shifted = dataset_utils.add_noise_to_selected_classes(
                train_dataset,
                selected_classes_noise,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
            )
            train_loader = DataLoader(
                train_dataset_shifted,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        elif condition == "all_perturbations":
            # 同时删除样本并注入噪声
            modified_dataset = dataset_utils.modify_dataset(
                dataset=train_dataset,
                selected_classes_remove=selected_classes_remove,
                selected_classes_noise=selected_classes_noise,
                remove_fraction=remove_fraction,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
            )
            train_loader = DataLoader(
                modified_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )
        else:
            raise ValueError("Unsupported condition: " + condition)

        pbar.update(1)  # 数据集加载完成

    # 创建模型
    print(f"Creating model: {model_name}")
    model_utils = ModelUtils(
        model_name, num_classes_dict[dataset_name], pretrained=pretrained
    )
    model = model_utils.create_model()

    # 选择优化器类型并设置相应的参数
    if optimizer == "sgd":
        criterion, optimizer = model_utils.get_criterion_and_optimizer(
            model,
            learning_rate=learning_rate,
            optimizer="sgd",
            momentum=momentum,  # 使用传递的 momentum 参数
            weight_decay=weight_decay,  # 使用传递的 weight decay 参数
        )
    elif optimizer == "adam":
        criterion, optimizer = model_utils.get_criterion_and_optimizer(
            model,
            learning_rate=learning_rate,
            optimizer="adam",
            weight_decay=weight_decay,  # 即使在 Adam 中也可以使用 weight decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # 训练和测试工具
    train_test_utils = TrainTestUtils(model_name, dataset_name)
    save_path = train_test_utils.create_save_path(condition)

    best_accuracy = 0
    patience_counter = 0

    alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 0.1)

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
            alpha=alpha,  # 传递 alpha 参数
            beta=beta,  # 传递 beta 参数
        )

        # 在验证集上评估模型性能
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        accuracy = train_test_utils.test(model, val_loader, condition)

        # 如果使用了 early stopping
        if use_early_stopping:
            # 检查是否达到了早停的条件
            if accuracy >= early_stopping_accuracy_threshold:
                print(
                    f"Accuracy {accuracy*100:.2f}% reached threshold. Stopping early."
                )
                break

            # 检查性能是否下降
            if accuracy < best_accuracy:
                patience_counter += 1  # 如果性能下降，计数器加 1
                print(
                    f"Performance dropped. Patience counter: {patience_counter}/{early_stopping_patience}"
                )
            elif accuracy > best_accuracy:
                # 如果准确率提高，更新 best_accuracy 并重置 patience_counter
                best_accuracy = accuracy
                patience_counter = 0
            else:
                # 性能持平时也重置计数器（持平不算下降）
                patience_counter = 0

            # 如果连续下降次数达到 early stopping 条件，则提前停止训练
            if patience_counter >= early_stopping_patience:
                print(
                    f"Performance dropped for {early_stopping_patience} consecutive epochs. Stopping early."
                )
                break
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed without early stopping.")

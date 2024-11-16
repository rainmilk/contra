import torch
from dataset_loader import DatasetLoader
from model_utils import ModelUtils
from train_test_utils import TrainTestUtils
from torch.utils.data import DataLoader

# 数据集路径配置
dataset_paths = {
    "cifar-10": "./data/cifar-10",
    "cifar-100": "./data/cifar-100",
    "animals-10": "./data/animals-10",
    "flowers-102": "./data/flowers-102",
}

# 类别数量
num_classes_dict = {
    "cifar-10": 10,
    "cifar-100": 100,
    "animals-10": 10,
    "flowers-102": 102,
}


# 创建 DatasetLoader 实例
def test_dataset(dataset_name):
    print(f"\nTesting dataset: {dataset_name}")

    # 加载数据集
    loader = DatasetLoader(dataset_name, dataset_paths, num_classes_dict)
    train_dataset, test_dataset = loader.get_dataset()

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # 创建 ResNet 模型
    model = ModelUtils.create_model(num_classes=num_classes_dict[dataset_name])

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型（进行少量 epoch 以便快速测试）
    TrainTestUtils.train_and_save(
        model,
        train_loader,
        criterion,
        optimizer,
        save_path=f"./models/{dataset_name}",
        num_epochs=1,
    )

    # 测试模型
    TrainTestUtils.test(model, test_loader)


# 测试所有数据集
if __name__ == "__main__":
    # for dataset in ["cifar-10", "cifar-100", "animals-10", "flowers-102"]:
    # for dataset in ["animals-10"]:
    for dataset in ["flowers-102"]:
        test_dataset(dataset)

import os
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_NAME = "resnet18"

RESCALE_SIZE = 256
CROP_SIZE = 224

if MODEL_NAME == "resnet18":
    RESCALE_SIZE = 256
    CROP_SIZE = 224


def generate_food101_cache(data_dir, gen_dir, batch_size=64, num_workers=4):

    # transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

    # 定义训练和测试集的 transforms

    data_transforms = transforms.Compose(
        [
            # transforms.Resize((96, 96)),
            transforms.Resize(RESCALE_SIZE),
            # transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 加载 FOOD-101 数据集
    print("Loading Food101 training and test datasets...")
    train_dataset = datasets.Food101(
        root=data_dir, split="train", transform=data_transforms
    )
    test_dataset = datasets.Food101(
        root=data_dir, split="test", transform=data_transforms
    )

    # 使用 DataLoader 进行批量加载
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 提取训练数据和标签并保存
    print("Extracting and saving training data and labels...")
    train_data, train_labels = [], []
    for inputs, labels in tqdm(train_loader):
        train_data.append(inputs)  # 保留为张量
        train_labels.append(labels)  # 保留为张量
    train_data = torch.cat(train_data, dim=0)  # 将数据连接起来
    train_labels = torch.cat(train_labels, dim=0)  # 将标签连接起来

    # 转换为 uint8 数据类型
    train_data_uint8 = (
        (train_data * 255).byte().numpy()
    )  # 将图像数据转为 uint8 并转换为 numpy 数组
    train_labels_uint8 = (
        train_labels.byte().numpy()
    )  # 将标签转为 uint8 并转换为 numpy 数组

    # 保存到 gen 目录，使用 numpy.save
    os.makedirs(gen_dir, exist_ok=True)
    np.save(os.path.join(gen_dir, "train_data.npy"), train_data_uint8)
    np.save(os.path.join(gen_dir, "train_labels.npy"), train_labels_uint8)

    # 提取测试数据和标签并保存
    print("Extracting and saving test data and labels...")
    test_data, test_labels = [], []
    for inputs, labels in tqdm(test_loader):
        test_data.append(inputs)  # 保留为张量
        test_labels.append(labels)  # 保留为张量
    test_data = torch.cat(test_data, dim=0)  # 将数据连接起来
    test_labels = torch.cat(test_labels, dim=0)  # 将标签连接起来

    # 转换为 uint8 数据类型
    test_data_uint8 = (
        (test_data * 255).byte().numpy()
    )  # 将图像数据转为 uint8 并转换为 numpy 数组
    test_labels_uint8 = (
        test_labels.byte().numpy()
    )  # 将标签转为 uint8 并转换为 numpy 数组

    # 保存到 gen 目录，使用 numpy.save
    np.save(os.path.join(gen_dir, "test_data.npy"), test_data_uint8)
    np.save(os.path.join(gen_dir, "test_labels.npy"), test_labels_uint8)

    print("Data caching complete.")


if __name__ == "__main__":
    data_dir = "./data/food-101/normal"  # 数据集目录
    gen_dir = "./data/food-101/gen/cache"  # 缓存生成目录
    generate_food101_cache(data_dir, gen_dir)

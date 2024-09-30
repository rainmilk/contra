import os
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def generate_food101_cache(data_dir, gen_dir, batch_size=512, num_workers=4):
    
    transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
   
    # 定义训练和测试集的 transforms
    # train_transforms = transforms.Compose([
    #     transforms.Resize((96, 96)),
    #     transforms.RandomRotation(30),
    #     transforms.RandomResizedCrop(224),
    #     # transforms.RandomHorizontalFlip(),
    #     # ImageNetPolicy(),  # 如果不需要可以去掉，或者换成其他增强方法
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # test_transforms = transforms.Compose([
    #     transforms.Resize((96, 96)),
    #     # transforms.Resize(255),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # 加载 FOOD-101 数据集
    print("Loading Food101 training and test datasets...")
    train_dataset = datasets.Food101(root=data_dir, split="train", transform=transform)
    test_dataset = datasets.Food101(root=data_dir, split="test", transform=transform)

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
        train_data.append(inputs.numpy())  # 转换为 numpy 数组
        train_labels.append(labels.numpy())  # 转换为 numpy 数组
    train_data = np.concatenate(train_data, axis=0)  # 将数据连接起来
    train_labels = np.concatenate(train_labels, axis=0)  # 将标签连接起来

    # 保存到 gen 目录
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    np.save(os.path.join(gen_dir, "train_data.npy"), train_data)
    np.save(os.path.join(gen_dir, "train_labels.npy"), train_labels)

    # 提取测试数据和标签并保存
    print("Extracting and saving test data and labels...")
    test_data, test_labels = [], []
    for inputs, labels in tqdm(test_loader):
        test_data.append(inputs.numpy())  # 转换为 numpy 数组
        test_labels.append(labels.numpy())  # 转换为 numpy 数组
    test_data = np.concatenate(test_data, axis=0)  # 将数据连接起来
    test_labels = np.concatenate(test_labels, axis=0)  # 将标签连接起来

    # 保存到 gen 目录
    np.save(os.path.join(gen_dir, "test_data.npy"), test_data)
    np.save(os.path.join(gen_dir, "test_labels.npy"), test_labels)
    print("Data caching complete.")


if __name__ == "__main__":
    data_dir = "./data/food-101/normal"  # 数据集目录
    gen_dir = "./data/food-101/gen"  # 缓存生成目录
    generate_food101_cache(data_dir, gen_dir)

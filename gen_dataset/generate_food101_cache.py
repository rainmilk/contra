import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate_food101_cache(data_dir, gen_dir, batch_size=512, num_workers=4):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # 加载 FOOD-101 数据集
    print("Loading Food101 training and test datasets...")
    train_dataset = datasets.Food101(root=data_dir, split="train", transform=transform)
    test_dataset = datasets.Food101(root=data_dir, split="test", transform=transform)

    # 使用 DataLoader 进行批量加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 提取训练数据和标签并保存
    print("Extracting and saving training data and labels...")
    train_data, train_labels = [], []
    for inputs, labels in tqdm(train_loader):
        train_data.append(inputs)
        train_labels.append(labels)
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)

    # 保存到 gen 目录
    os.makedirs(gen_dir, exist_ok=True)
    torch.save(train_data, os.path.join(gen_dir, "train_data.npy"))
    torch.save(train_labels, os.path.join(gen_dir, "train_labels.npy"))

    # 提取测试数据和标签并保存
    print("Extracting and saving test data and labels...")
    test_data, test_labels = [], []
    for inputs, labels in tqdm(test_loader):
        test_data.append(inputs)
        test_labels.append(labels)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)

    # 保存到 gen 目录
    torch.save(test_data, os.path.join(gen_dir, "test_data.npy"))
    torch.save(test_labels, os.path.join(gen_dir, "test_labels.npy"))

    print("Data caching complete.")

if __name__ == "__main__":
    data_dir = "./data/food-101/normal"  # 数据集目录
    gen_dir = "./data/food-101/gen"      # 缓存生成目录
    generate_food101_cache(data_dir, gen_dir)
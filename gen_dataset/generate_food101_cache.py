import os
from tqdm import tqdm
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_NAME = "resnet18"

RESCALE_SIZE = (224, 224)

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
        root=data_dir, split="train", transform=data_transforms, download=True
    )
    test_dataset = datasets.Food101(
        root=data_dir, split="test", transform=data_transforms, download=True
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
    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)

    # 保存到 gen 目录，使用 numpy.save
    os.makedirs(gen_dir, exist_ok=True)
    np.save(os.path.join(gen_dir, "train_data.npy"), train_data)
    np.save(os.path.join(gen_dir, "train_labels.npy"), train_labels)

    # 保存到 gen 目录，使用 numpy.save
    np.save(os.path.join(gen_dir, "test_data.npy"), test_data)
    np.save(os.path.join(gen_dir, "test_labels.npy"), test_labels)

    print("Data caching complete.")


if __name__ == "__main__":
    data_dir = "./data/food-101/normal"  # 数据集目录
    gen_dir = "./data/food-101/gen/cache"  # 缓存生成目录
    generate_food101_cache(data_dir, gen_dir)

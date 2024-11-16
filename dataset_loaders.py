"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100
        Flowers102
        TinyImageNet-200
          
"""

import os
import copy
import glob
from tqdm import tqdm
from shutil import move

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, Flowers102

import numpy as np
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, image_folders, transform=None):
        self.imgs = torch.stack(
            [transform(Image.open(img)) for img in tqdm(image_folders._image_files)]
        )
        self.targets = image_folders._labels
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        return (self.transform(img) if self.transform else img), self.targets[idx]


# 预处理并保存数据到指定目录
def preprocess_and_save_data(dataset, file_path):
    print(f"Preprocessing and saving data to {file_path}...")
    preprocessed_data = []

    # 对数据集中的每张图片进行预处理
    for sample in tqdm(dataset.imgs):
        img_path = sample[0]
        label = sample[1]
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)  # 将图片转为张量
        preprocessed_data.append((img_tensor, label))

    # 保存为 .pt 文件
    torch.save(preprocessed_data, file_path)
    print(f"Data saved to {file_path}.")
    return preprocessed_data


# 从缓存文件加载数据，并使用 weights_only=True 消除警告
def load_preprocessed_data(file_path):
    print(f"Loading preprocessed data from {file_path}...")
    return torch.load(file_path, weights_only=True)


# 修改后的 TinyImageNetDataset 类，缓存文件保存到指定目录
class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        image_folder_set,
        cache_file,
        norm_trans=None,
        start=0,
        end=-1,
    ):
        self.cache_file = cache_file  # 缓存文件路径应由调用方传入
        self.norm_trans = norm_trans

        # 检查缓存文件是否存在
        if os.path.exists(self.cache_file):
            # 如果缓存文件存在，加载预处理后的数据
            self.data = load_preprocessed_data(self.cache_file)
        else:
            # 否则预处理数据并保存到缓存文件
            self.imgs = image_folder_set.imgs[start:end]
            self.data = preprocess_and_save_data(self, self.cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.norm_trans:
            img = self.norm_trans(img)  # 进行归一化等转换
        return img, label


class TinyImageNet:
    """
    TinyImageNet dataset loader.
    """

    def __init__(self, data_dir, normalize=False):
        self.data_dir = data_dir
        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )
        self.tr_train = transforms.Compose(
            [transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()]
        )
        self.tr_test = transforms.Compose(
            []
        )  # You can add test-time transformations here
        self.train_path = os.path.join(self.data_dir, "train/")
        self.val_path = os.path.join(self.data_dir, "val/")
        self.test_path = os.path.join(self.data_dir, "test/")
        self._organize_val_test()

    def _organize_val_test(self):
        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(self.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {
                line.split("\t")[0]: line.split("\t")[1]
                for line in open(
                    os.path.join(self.val_path, "val_annotations.txt")
                ).readlines()
            }
            paths = glob.glob(os.path.join(self.val_path, "images/*"))
            for path in paths:
                folder = val_dict[path.split("/")[-1]]
                dest_dir = os.path.join(
                    (
                        self.test_path
                        if len(
                            glob.glob(
                                os.path.join(self.val_path, folder, "images", "*")
                            )
                        )
                        >= 25
                        else self.val_path
                    ),
                    folder,
                    "images",
                )
                os.makedirs(dest_dir, exist_ok=True)
                move(path, os.path.join(dest_dir, os.path.basename(path)))
            os.rmdir(os.path.join(self.val_path, "images"))


def create_dataloaders(
    dataset_cls,
    data_dir,
    batch_size,
    num_workers,
    seed,
    train_transform,
    test_transform,
    val_ratio=0.1,
):
    dataset = dataset_cls(
        data_dir, train=True, transform=train_transform, download=True
    )
    test_set = dataset_cls(
        data_dir, train=False, transform=test_transform, download=True
    )

    train_set, val_set = split_dataset(dataset, seed, val_ratio)
    return (
        create_loader(train_set, batch_size, num_workers, seed),
        create_loader(val_set, batch_size, num_workers, seed, shuffle=False),
        create_loader(test_set, batch_size, num_workers, seed, shuffle=False),
    )


def split_dataset(dataset, seed, val_ratio):
    np.random.seed(seed)
    labels = np.unique(dataset.targets)
    valid_idx = [
        idx
        for label in labels
        for idx in np.where(np.array(dataset.targets) == label)[0][
            : int(val_ratio * len(np.where(np.array(dataset.targets) == label)[0]))
        ]
    ]
    train_idx = list(set(range(len(dataset))) - set(valid_idx))

    valid_set = copy.deepcopy(dataset)
    valid_set.data = dataset.data[valid_idx]
    valid_set.targets = np.array(dataset.targets)[valid_idx]

    train_set = copy.deepcopy(dataset)
    train_set.data = dataset.data[train_idx]
    train_set.targets = np.array(dataset.targets)[train_idx]

    return train_set, valid_set


def create_loader(dataset, batch_size, num_workers, seed, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(seed),
    )


def cifar10_dataloaders(
    batch_size=128, data_dir="data/cifar-10", num_workers=2, seed=1
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return create_dataloaders(
        CIFAR10,
        data_dir,
        batch_size,
        num_workers,
        seed,
        train_transform,
        test_transform,
    )


def cifar100_dataloaders(
    batch_size=128, data_dir="data/cifar-100", num_workers=2, seed=1
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ]
    )

    return create_dataloaders(
        CIFAR100,
        data_dir,
        batch_size,
        num_workers,
        seed,
        train_transform,
        test_transform,
    )


# Helper function to get the Flowers102 dataset for train or val split
def get_flowers_train_or_val(root_dir: str, split: str):
    return torchvision.datasets.Flowers102(
        root=root_dir,
        download=True,
        split=split,
        transform=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
    )


# 获取 Flowers102 数据集
def get_flowers_102_dataset(root_dir: str):
    # 加载训练集 (我们使用 'test' 作为训练集并应用数据增强)
    flowers_train_dataset = torchvision.datasets.Flowers102(
        root=root_dir,
        split="test",
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3),
                torchvision.transforms.RandomAffine(degrees=30, shear=20),
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ]
        ),
    )

    flowers_val_dataset = get_flowers_train_or_val(root_dir, "val")

    flowers_test_dataset = get_flowers_train_or_val(root_dir, "train")

    return flowers_train_dataset, flowers_val_dataset, flowers_test_dataset


# DataLoader function for Flowers102
def Flowers102_dataloaders(
    batch_size=128, data_dir="data/flowers-102", num_workers=2, seed=1
):
    # 获取训练集和测试集
    train_set, val_set, test_set = get_flowers_102_dataset(data_dir)

    # 打印数据集大小
    print(f"Training set size: {len(train_set)}, Test set size: {len(test_set)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def tinyImageNet_dataloaders(
    batch_size=128,
    data_dir="data/tiny-imagenet-200",
    num_workers=2,
    pin_memory=True,  # 如果使用GPU，开启pin_memory
    seed=1,
    normalize=False,
):
    # 创建TinyImageNet实例，并直接传递data_dir参数
    tiny_imagenet = TinyImageNet(data_dir=data_dir, normalize=normalize)

    # 定义缓存文件路径，保存到data_dir下
    train_cache_file = os.path.join(data_dir, "tiny_imagenet_train_cache.pt")
    val_cache_file = os.path.join(data_dir, "tiny_imagenet_val_cache.pt")
    test_cache_file = os.path.join(data_dir, "tiny_imagenet_test_cache.pt")

    # 加载训练集
    train_set = ImageFolder(tiny_imagenet.train_path, transform=tiny_imagenet.tr_train)
    train_loader = DataLoader(
        TinyImageNetDataset(
            train_set, cache_file=train_cache_file, norm_trans=tiny_imagenet.norm_layer
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            pin_memory if torch.cuda.is_available() else False
        ),  # 仅在有GPU时启用 pin_memory
        worker_init_fn=lambda _: np.random.seed(seed),
    )

    # 加载测试集
    test_set = ImageFolder(tiny_imagenet.test_path, transform=tiny_imagenet.tr_test)
    test_loader = DataLoader(
        TinyImageNetDataset(
            test_set, cache_file=test_cache_file, norm_trans=tiny_imagenet.norm_layer
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda _: np.random.seed(seed),
    )

    # 加载验证集
    val_set = ImageFolder(tiny_imagenet.val_path, transform=tiny_imagenet.tr_test)
    val_loader = DataLoader(
        TinyImageNetDataset(
            val_set, cache_file=val_cache_file, norm_trans=tiny_imagenet.norm_layer
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda _: np.random.seed(seed),
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # CIFAR-10 Dataloaders
    print("Testing CIFAR-10 Dataloaders...")
    train_loader, val_loader, test_loader = cifar10_dataloaders()

    print(f"Number of images in training set: {len(train_loader.dataset)}")
    print(f"Number of images in validation set: {len(val_loader.dataset)}")
    print(f"Number of images in test set: {len(test_loader.dataset)}")

    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in validation loader: {len(val_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")

    for i, (img, label) in enumerate(train_loader):
        if i == 0:
            print(
                f"First batch of train loader - image shape: {img.shape}, label shape: {label.shape}"
            )
        print(
            f"Batch {i+1} in train loader - unique labels: {torch.unique(label)}, batch size: {len(label)}"
        )
        if i == 1:  # Only print details for the first 2 batches for brevity
            break

    # CIFAR-100 Dataloaders
    print("\nTesting CIFAR-100 Dataloaders...")
    train_loader, val_loader, test_loader = cifar100_dataloaders()

    print(f"Number of images in training set: {len(train_loader.dataset)}")
    print(f"Number of images in validation set: {len(val_loader.dataset)}")
    print(f"Number of images in test set: {len(test_loader.dataset)}")

    print(f"Number of batches in CIFAR-100 train loader: {len(train_loader)}")
    print(f"Number of batches in CIFAR-100 validation loader: {len(val_loader)}")
    print(f"Number of batches in CIFAR-100 test loader: {len(test_loader)}")

    for i, (img, label) in enumerate(train_loader):
        print(
            f"Batch {i+1} in CIFAR-100 train loader - unique labels: {torch.unique(label)}, batch size: {len(label)}"
        )
        if i == 1:
            break

    # Flowers102 Dataloaders
    print("\nTesting Flowers102 Dataloaders...")
    train_loader, val_loader, test_loader = Flowers102_dataloaders()

    print(f"Number of images in training set: {len(train_loader.dataset)}")
    print(f"Number of images in validation set: {len(val_loader.dataset)}")
    print(f"Number of images in test set: {len(test_loader.dataset)}")

    print(f"Number of batches in Flowers-102 train loader: {len(train_loader)}")
    print(f"Number of batches in Flowers-102 validation loader: {len(val_loader)}")
    print(f"Number of batches in Flowers-102 test loader: {len(test_loader)}")

    for i, (img, label) in enumerate(train_loader):
        print(
            f"Batch {i+1} in Flowers102 train loader - unique labels: {torch.unique(label)}, batch size: {len(label)}"
        )
        if i == 1:
            break

    # TinyImageNet Dataloaders
    print("\nTesting TinyImageNet Dataloaders...")
    train_loader, val_loader, test_loader = tinyImageNet_dataloaders()

    print(f"Number of images in training set: {len(train_loader.dataset)}")
    print(f"Number of images in validation set: {len(val_loader.dataset)}")
    print(f"Number of images in test set: {len(test_loader.dataset)}")

    print(f"Number of batches in TinyImageNet train loader: {len(train_loader)}")
    print(f"Number of batches in TinyImageNet validation loader: {len(val_loader)}")
    print(f"Number of batches in TinyImageNet test loader: {len(test_loader)}")

    for i, (img, label) in enumerate(train_loader):
        print(
            f"Batch {i+1} in TinyImageNet train loader - unique labels: {torch.unique(label)}, batch size: {len(label)}"
        )
        if i == 1:
            break

import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset

from utils.dataset_animals_10 import get_animals_10_dataset
from utils.dataset_flowers_102 import get_flowers_102_dataset

import pandas as pd
from tabulate import tabulate


# Custom loader for ImageFolder
def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class DatasetLoader:
    def __init__(self, dataset_name, dataset_paths, num_classes_dict):
        self.dataset_name = dataset_name
        self.dataset_paths = dataset_paths
        self.num_classes = num_classes_dict[dataset_name]

    def get_dataset(self):
        if self.dataset_name == "cifar-10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.dataset_paths[self.dataset_name],
                train=True,
                download=True,
                transform=transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.dataset_paths[self.dataset_name],
                train=False,
                download=True,
                transform=transform,
            )
        elif self.dataset_name == "cifar-100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.dataset_paths[self.dataset_name],
                train=True,
                download=True,
                transform=transform,
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.dataset_paths[self.dataset_name],
                train=False,
                download=True,
                transform=transform,
            )
        elif self.dataset_name == "animals-10":
            train_dataset, test_dataset = get_animals_10_dataset(
                self.dataset_paths[self.dataset_name]
            )
        elif self.dataset_name == "flowers-102":
            train_dataset, test_dataset = get_flowers_102_dataset(
                self.dataset_paths[self.dataset_name]
            )
        else:
            raise ValueError("Unsupported dataset: " + self.dataset_name)
        return train_dataset, test_dataset

    # 操作1：删除选定类别中的指定比例样本
    def remove_fraction_of_selected_classes(
        self, dataset, selected_classes, remove_fraction=0.5
    ):
        class_indices = {i: [] for i in selected_classes}

        # 为每个选定的类别找到对应的样本索引
        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes:
                class_indices[label].append(idx)

        removed_indices = []
        # 随机删除指定比例的样本
        for label in selected_classes:
            indices = class_indices[label]
            removed_indices.extend(
                random.sample(indices, int(len(indices) * remove_fraction))
            )

        remaining_indices = list(set(range(len(dataset))) - set(removed_indices))
        return Subset(dataset, remaining_indices)

    # 操作2：为选定类别添加噪声
    def add_noise_to_selected_classes(
        self, dataset, selected_classes, noise_fraction=0.1
    ):
        noisy_data = []
        noisy_labels = []

        for idx, (image, label) in enumerate(dataset):
            if label in selected_classes and random.random() < noise_fraction:
                noise = torch.randn_like(image) * 0.1  # 添加噪声
                image = image + noise
                image = torch.clamp(image, -1, 1)  # 确保图像仍在合法范围内
            noisy_data.append(image)
            noisy_labels.append(label)

        return list(zip(noisy_data, noisy_labels))

    # 组合操作：同时删除样本并注入噪声
    def modify_dataset(
        self,
        dataset,
        selected_classes_remove,
        selected_classes_noise,
        remove_fraction=0.5,
        noise_fraction=0.1,
    ):
        """
        同时执行删除样本和注入噪声的操作
        :param dataset: 原始数据集
        :param selected_classes_remove: 要删除样本的类别列表
        :param selected_classes_noise: 要注入噪声的类别列表
        :param remove_fraction: 删除样本的比例
        :param noise_fraction: 注入噪声的比例
        """
        # 操作1：删除选定类别中的指定比例样本
        dataset_after_removal = self.remove_fraction_of_selected_classes(
            dataset, selected_classes_remove, remove_fraction
        )

        # 操作2：为剩余类别添加噪声
        noisy_dataset = self.add_noise_to_selected_classes(
            dataset_after_removal, selected_classes_noise, noise_fraction
        )

        return noisy_dataset

    # 统计数据分布
    def compute_statistics(self, dataset):
        class_counts = {}
        pixel_means = []
        pixel_stds = []

        # 计算每个类别的样本数，以及像素的均值和标准差
        for image, label in dataset:
            # 统计类别数
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

            # 计算像素均值和标准差
            pixel_means.append(image.mean().item())
            pixel_stds.append(image.std().item())

        return (
            class_counts,
            sum(pixel_means) / len(pixel_means),
            sum(pixel_stds) / len(pixel_stds),
        )


# 使用例子
if __name__ == "__main__":
    dataset_loader = DatasetLoader(
        dataset_name="cifar-10",
        dataset_paths={
            "cifar-10": "./data/cifar-10",
            "cifar-100": "./data/cifar-100",
            "animals-10": "./data/animals-10",
            "flowers-102": "./data/flowers-102",
        },
        num_classes_dict={
            "cifar-10": 10,
            "cifar-100": 100,
            "animals-10": 10,
            "flowers-102": 102,
        },
    )

    train_dataset, _ = dataset_loader.get_dataset()

    # CIFAR-10操作例子
    modified_dataset = dataset_loader.modify_dataset(
        dataset=train_dataset,
        selected_classes_remove=[0, 1, 2, 3, 4],  # 删除类别 0 到 4
        selected_classes_noise=[5, 6, 7, 8, 9],  # 对类别 5 到 9 注入噪声
        remove_fraction=0.5,  # 删除 50% 的样本
        noise_fraction=0.1,  # 注入 10% 的噪声
    )

    # 计算并显示修改前后的数据分布
    original_class_counts, original_mean, original_std = (
        dataset_loader.compute_statistics(train_dataset)
    )
    modified_class_counts, modified_mean, modified_std = (
        dataset_loader.compute_statistics(modified_dataset)
    )

    # 打印统计量对比
    data = {
        "Metric": ["Class Distribution", "Pixel Mean", "Pixel Std"],
        "Original": [original_class_counts, original_mean, original_std],
        "Modified": [modified_class_counts, modified_mean, modified_std],
    }
    df = pd.DataFrame(data)
    print(tabulate(df, headers="keys", tablefmt="grid"))

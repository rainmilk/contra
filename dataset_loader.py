import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset
from utils.dataset_animals_10 import get_animals_10_dataset
from utils.dataset_flowers_102 import get_flowers_102_dataset
import pandas as pd
from tabulate import tabulate
from PIL import Image


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
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            )
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.dataset_paths[self.dataset_name],
                train=True,
                download=True,
                transform=transform_train,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.dataset_paths[self.dataset_name],
                train=False,
                download=True,
                transform=transform_test,
            )
        elif self.dataset_name == "cifar-100":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
                    ),
                ]
            )
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.dataset_paths[self.dataset_name],
                train=True,
                download=True,
                transform=transform_train,
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.dataset_paths[self.dataset_name],
                train=False,
                download=True,
                transform=transform_test,
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

    # 扁平化类标签列表，处理嵌套列表
    def flatten_class_list(self, class_list):
        flattened = []
        for item in class_list:
            if isinstance(item, list):
                flattened.extend(self.flatten_class_list(item))  # 递归展开列表
            else:
                flattened.append(int(item))
        return flattened

    # 操作1：删除选定类别中的指定比例样本
    def remove_fraction_of_selected_classes(
        self, dataset, selected_classes, remove_fraction=0.5
    ):
        # 将类标签展平为一维列表
        selected_classes = self.flatten_class_list(selected_classes)

        # 打印检查selected_classes是否正确
        print(type(selected_classes))
        print(selected_classes)

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
        self, dataset, selected_classes, noise_fraction=0.8, noise_type="gaussian"
    ):
        # 扁平化类标签
        selected_classes = self.flatten_class_list(selected_classes)

        noisy_data = []
        noisy_labels = []

        for idx, (image, label) in enumerate(dataset):
            if label in selected_classes and random.random() < noise_fraction:
                if noise_type == "gaussian":
                    noise = torch.randn_like(image) * 0.1  # 高斯噪声
                    image = image + noise
                    image = torch.clamp(image, -1, 1)  # 确保图像仍在合法范围内
                elif noise_type == "salt_pepper":
                    image = self.add_salt_and_pepper_noise(image)
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")
            noisy_data.append(image)
            noisy_labels.append(label)

        return list(zip(noisy_data, noisy_labels))

    # 椒盐噪声的辅助函数
    def add_salt_and_pepper_noise(self, image, amount=0.05, salt_vs_pepper=0.5):
        """
        椒盐噪声
        :param image: 输入的图像张量
        :param amount: 噪声量的比例
        :param salt_vs_pepper: 盐和胡椒的比例
        """
        noisy_image = image.clone()
        num_salt = int(amount * image.numel() * salt_vs_pepper)
        num_pepper = int(amount * image.numel() * (1.0 - salt_vs_pepper))

        # 添加盐噪声
        coords = [torch.randint(0, i, (num_salt,)) for i in image.shape]
        noisy_image[coords] = 1

        # 添加胡椒噪声
        coords = [torch.randint(0, i, (num_pepper,)) for i in image.shape]
        noisy_image[coords] = 0

        return noisy_image

    # 组合操作：同时删除样本并注入噪声
    def modify_dataset(
        self,
        dataset,
        selected_classes_remove,
        selected_classes_noise,
        remove_fraction=0.5,
        noise_fraction=0.1,
        noise_type="gaussian",
    ):
        """
        同时执行删除样本和注入噪声的操作
        :param dataset: 原始数据集
        :param selected_classes_remove: 要删除样本的类别列表
        :param selected_classes_noise: 要注入噪声的类别列表
        :param remove_fraction: 删除样本的比例
        :param noise_fraction: 注入噪声的比例
        :param noise_type: 噪声类型（高斯或椒盐）
        """
        # 操作1：删除选定类别中的指定比例样本
        dataset_after_removal = self.remove_fraction_of_selected_classes(
            dataset, selected_classes_remove, remove_fraction
        )

        # 操作2：为剩余类别添加噪声
        noisy_dataset = self.add_noise_to_selected_classes(
            dataset_after_removal, selected_classes_noise, noise_fraction, noise_type
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
        noise_fraction=0.5,  # 注入 50% 的噪声
        noise_type="salt_pepper",  # 使用椒盐噪声
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

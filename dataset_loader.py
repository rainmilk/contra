import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset


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
        else:
            raise ValueError("Unsupported dataset: " + self.dataset_name)

        return train_dataset, test_dataset

    def remove_50_percent_of_selected_classes(
        self, dataset, selected_classes, remove_fraction=0.5
    ):
        class_indices = {i: [] for i in selected_classes}

        for idx, (_, label) in enumerate(dataset):
            if label in selected_classes:
                class_indices[label].append(idx)

        removed_indices = []
        for label in selected_classes:
            indices = class_indices[label]
            removed_indices.extend(
                random.sample(indices, int(len(indices) * remove_fraction))
            )

        remaining_indices = list(set(range(len(dataset))) - set(removed_indices))
        return Subset(dataset, remaining_indices)

    def add_noise_to_selected_classes(
        self, dataset, selected_classes, noise_fraction=0.1
    ):
        noisy_data = []
        noisy_labels = []

        for idx, (image, label) in enumerate(dataset):
            if label in selected_classes and random.random() < noise_fraction:
                noise = torch.randn_like(image) * 0.1
                image = image + noise
                image = torch.clamp(image, -1, 1)
            noisy_data.append(image)
            noisy_labels.append(label)

        return list(zip(noisy_data, noisy_labels))

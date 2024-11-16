"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100
        FashionMNIST
        SVHN
        Flowers102
            
"""

import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    Flowers102,
    ImageFolder,
)
from tqdm import tqdm


def cifar10_dataloaders(
    batch_size=128,
    data_dir="data/cifar10",
    num_workers=2,
    class_to_replace: str = None,  #
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    #  固定 valid
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_num = int(0.1 * len(class_idx))
        valid_idx.append(class_idx[:valid_num])
        # valid_idx.append(
        #     rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        # )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            #  效率低
            if len(class_to_replace) > 1:
                class_replace_list = class_to_replace.split(",")
                class_replace_list = [int(data) for data in class_replace_list]
                replace_idxes = []
                for class_replace in class_replace_list:
                    replace_idx = np.where(test_set.targets == class_replace)[0]
                    replace_idxes.extend(replace_idx)

                retrain_idx = [
                    id for id in range(len(test_set)) if id not in replace_idxes
                ]
                test_set.data = test_set.data[retrain_idx]
                test_set.targets = test_set.targets[retrain_idx]
            else:
                class_to_replace = int(class_to_replace)
                test_set.data = test_set.data[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="data/cifar100",
    num_workers=2,
    class_to_replace: str = None,  #
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        #
        valid_num = int(0.1 * len(class_idx))
        valid_idx.append(class_idx[:valid_num])
        # valid_idx.append(
        #     rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        # )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 450:
            #
            if len(class_to_replace) > 1:
                class_replace_list = class_to_replace.split(",")
                class_replace_list = [int(data) for data in class_replace_list]
                replace_idxes = []
                for class_replace in class_replace_list:
                    replace_idx = np.where(test_set.targets == class_replace)[0]
                    replace_idxes.extend(replace_idx)

                retrain_idx = [
                    id for id in range(len(test_set)) if id not in replace_idxes
                ]
                test_set.data = test_set.data[retrain_idx]
                test_set.targets = test_set.targets[retrain_idx]
            else:
                class_to_replace = int(class_to_replace)
                test_set.data = test_set.data[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


class flowers102_dataset(Dataset):
    def __init__(self, image_folders):
        self.imgs = []
        self.targets = image_folders._labels
        self.transform = image_folders.transform
        for sample in tqdm(image_folders._image_files):
            img = self.transform(Image.open(sample))
            # img = Image.open(sample)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]


def flowers102_dataloaders(
    batch_size=128,
    data_dir="data/flowers102",
    num_workers=2,
    class_to_replace: str = None,  #  int -> str
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: Flower102\t 1020 images for training \t 1020 images for validation\t"
    )
    print("6149 images for testing\t no normalize applied in data_transform")

    train_folders = Flowers102(
        data_dir, transform=data_transform, split="train", download=True
    )
    train_set = flowers102_dataset(train_folders)
    train_set.targets = np.array(train_set.targets)
    train_set.targets = np.array(train_set.targets)

    if class_to_replace:
        test_folders = Flowers102(
            data_dir, transform=data_transform, split="test", download=True
        )
        test_set = flowers102_dataset(test_folders)
    else:
        valid_folders = Flowers102(
            data_dir, transform=data_transform, split="val", download=True
        )
        valid_set = flowers102_dataset(valid_folders)

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 10:
            #
            if len(class_to_replace) > 1:
                class_replace_list = class_to_replace.split(",")
                class_replace_list = [int(data) for data in class_replace_list]
                replace_idxes = []
                for class_replace in class_replace_list:
                    replace_idx = np.where(test_set.targets == class_replace)[0]
                    replace_idxes.extend(replace_idx)

                retrain_idx = [
                    id for id in range(len(test_set)) if id not in replace_idxes
                ]
                test_set.data = test_set.data[retrain_idx]
                test_set.targets = test_set.targets[retrain_idx]
            else:
                class_to_replace = int(class_to_replace)
                test_set.data = test_set.data[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    if class_to_replace:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        return train_loader, None, test_loader
    else:
        val_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        return train_loader, val_loader, None

class TinyImageNetDataset:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))


def tinyImageNet_dataloaders(
    self,
    batch_size=128,
    data_dir="data/tiny-imagenet-200",
    num_workers=2,
    class_to_replace: str = None,  #
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    train_set = ImageFolder(self.train_path, transform=self.tr_train)
    train_set = TinyImageNetDataset(train_set, self.norm_layer)
    test_set = ImageFolder(self.test_path, transform=self.tr_test)
    test_set = TinyImageNetDataset(test_set, self.norm_layer)
    train_set.targets = np.array(train_set.targets)
    train_set.targets = np.array(train_set.targets)
    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.imgs = train_set_copy.imgs[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.imgs = train_set_copy.imgs[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 500:
            #
            test_set.targets = np.array(test_set.targets)
            if len(class_to_replace) > 1:
                class_replace_list = class_to_replace.split(",")
                class_replace_list = [int(data) for data in class_replace_list]
                replace_idxes = []
                for class_replace in class_replace_list:
                    replace_idx = np.where(test_set.targets == class_replace)[0]
                    replace_idxes.extend(replace_idx)

                retrain_idx = [
                    id for id in range(len(test_set)) if id not in replace_idxes
                ]
                test_set.imgs = test_set.imgs[retrain_idx]
                test_set.targets = test_set.targets[retrain_idx]
            else:
                class_to_replace = int(class_to_replace)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
            print(test_set.targets)
            test_set.targets = test_set.targets.tolist()
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,  #  tinyImagenet 原始为test_set?
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    print(
        f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
    )
    return train_loader, val_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def get_replace_classes(dataset: torch.utils.data.Dataset, class_to_replace: str):
    if len(class_to_replace) == 2 and int(class_to_replace) == -1:
        return class_to_replace
    elif len(class_to_replace) > 1:
        if "-" in class_to_replace:  # start;interval;num
            class_start, interval, list_num = class_to_replace.split("-")
            try:
                label_max = max(set(dataset.targets))
            except:
                try:
                    label_max = max(set(dataset.labels))
                except:
                    label_max = max(set(dataset._labels))

            class_replace_list = [
                str(i) for i in range(int(class_start), label_max, int(interval))
            ]
            class_replace_list = class_replace_list[: int(list_num)]
        else:
            class_replace_list = class_to_replace.split(",")
        return class_replace_list
    else:
        return class_to_replace


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: str,  #  int->str
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if len(class_to_replace) == 2 and int(class_to_replace) == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    #  add forget multiple classes 1 3 5
    elif len(class_to_replace) > 1:
        if "-" in class_to_replace:  # start; interval;num
            class_start, interval, list_num = class_to_replace.split("-")
            try:
                label_max = max(set(dataset.targets))
            except:
                try:
                    label_max = max(set(dataset.labels))
                except:
                    label_max = max(set(dataset._labels))

            class_replace_list = [
                i for i in range(int(class_start), label_max, int(interval))
            ]
            class_replace_list = class_replace_list[: int(list_num)]
        else:
            class_replace_list = class_to_replace.split(",")
        indexes_list = []
        for class_replace in class_replace_list:
            class_replace = int(class_replace)
            try:
                indexes = np.flatnonzero(np.array(dataset.targets) == class_replace)
            except:
                try:
                    indexes = np.flatnonzero(np.array(dataset.labels) == class_replace)
                except:
                    indexes = np.flatnonzero(np.array(dataset._labels) == class_replace)
            indexes_list.append(indexes)
    else:
        class_to_replace = int(class_to_replace)
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        if isinstance(class_to_replace, int) or (
            len(class_to_replace) == 2 and int(class_to_replace) == -1
        ):
            assert num_indexes_to_replace <= len(
                indexes
            ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
            rng = np.random.RandomState(seed)
            #  改为固定规则
            if num_indexes_to_replace <= 1.0:
                num_indexes_to_replace = round(num_indexes_to_replace * len(indexes))
            else:
                num_indexes_to_replace = int(num_indexes_to_replace)

            idx = np.arange(0, len(indexes), len(indexes) // num_indexes_to_replace)
            indexes = indexes[idx]
            indexes = indexes[:num_indexes_to_replace]

            print(f"Replacing indexes {indexes}")
        else:
            indexes_len = len(indexes_list[0])
            assert (
                num_indexes_to_replace <= indexes_len
            ), f"Want to replace {num_indexes_to_replace} indexes but only {indexes_len} samples in dataset"

            r_indexes = []
            if num_indexes_to_replace <= 1.0:
                num_indexes_to_replace = round(num_indexes_to_replace * len(indexes))
            else:
                num_indexes_to_replace = int(num_indexes_to_replace)

            for indexes in indexes_list:
                idx = np.arange(0, len(indexes), len(indexes) // num_indexes_to_replace)
                indexes = indexes[idx]
                indexes = indexes[:num_indexes_to_replace]
                r_indexes.append(indexes)
            print(f"Replacing indexes {r_indexes}")
            indexes = np.concatenate(r_indexes)
    replace_indexes(dataset, indexes, seed, only_mark)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)

import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from configs import settings
from .dataset import BaseTensorDataset


class MixupDataset(Dataset):
    def __init__(
        self,
        data_pair,
        label_pair,
        mixup_alpha=0.2,
        transforms=None,
        first_max=True
    ):
        # modify shape to [N, H, W, C]
        self.data_first = data_pair[0]
        self.data_second = data_pair[1]
        self.label_first = label_pair[0]
        self.label_second = label_pair[1]
        self.nb_second = len(self.label_second)
        self.second_idx = 0
        self.mixup_alpha = mixup_alpha
        self.transforms = transforms
        self.first_max = first_max

    def __len__(self):
        return len(self.label_first)

    def __getitem__(self, index):
        if self.mixup_alpha >= 0:
            lbd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if self.first_max and lbd < 0.5:
                lbd = 1 - lbd
        else:
            lbd = np.random.beta(-self.mixup_alpha, -self.mixup_alpha)
            if self.first_max and lbd > 0.5:
                lbd = 1 - lbd

        data_first = self.data_first[index]
        label_first = self.label_first[index]
        rnd_idx = self.second_idx % self.nb_second
        self.second_idx += 1
        data_rnd_ax = self.data_second[rnd_idx]
        label_rnd_ax = self.label_second[rnd_idx]

        if self.transforms is not None:
            data_first = self.transforms(data_first)
            data_rnd_ax = self.transforms(data_rnd_ax)

        mixed_data = lbd * data_first + (1 - lbd) * data_rnd_ax
        mixed_labels = lbd * label_first + (1 - lbd) * label_rnd_ax
        return mixed_data, mixed_labels


def get_dataset_loader(
    dataset_name,
    loader_name,
    case,
    step=None,
    batch_size=64,
    num_classes=0,
    drop_last=False,
    shuffle=False,
    onehot_enc=False,
    transforms=None,
    num_workers=0,
):
    """
    根据 loader_name 加载相应的数据集：支持增量训练 (inc)、辅助数据 (aux) 、测试数据 (test)和 D0数据集(train)
    """
    if not isinstance(loader_name, (list, tuple)):
        loader_name = [loader_name]

    if not isinstance(case, (list, tuple)):
        case = [case] * len(loader_name)


    data = []
    labels = []
    for ld_name, case_name in zip(loader_name, case):
        data_name = f"{ld_name}_data"
        data_path = settings.get_dataset_path(
            dataset_name, case_name, data_name, step
        )
        label_name = f"{ld_name}_label"
        label_path = settings.get_dataset_path(
            dataset_name, case_name, label_name, step
        )

        print(f"Loading {data_path}")

        data.append(np.load(data_path))
        label = np.load(label_path)
        labels.append(label.astype(np.int64))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # if loader_name == "train":
    #     transform = True

    if onehot_enc:  # train label change to onehot for teacher model
        labels = np.eye(num_classes)[labels]

    # 构建自定义数据集
    dataset = BaseTensorDataset(data, labels, transforms=transforms)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers
    )

    return data, labels, data_loader


def random_crop(img, img_size, padding=4):
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "constant")
    h, w = img.shape[1:]

    new_h, new_w = img_size
    start_x = np.random.randint(0, w - new_w)
    start_y = np.random.randint(0, h - new_h)

    crop_img = img[start_y : start_y + new_h, start_x : start_x + new_w]
    return crop_img


def random_horiz_flip(img):
    if random.random() > 0.5:
        img = np.fliplr(img)
    return img


if __name__ == "__main__":
    # 假设你的 CIFAR-10 数据存储在这个目录
    data_dir = "./data/cifar-10/noise/"
    # data_dir = "../data/cifar-100/noise/"
    # data_dir = "../data/tiny-imagenet-200/noise/"
    # data_dir = "../data/flowers-102/noise/"
    batch_size = 32


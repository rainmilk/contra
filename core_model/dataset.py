import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from configs import settings


class BaseTensorDataset(Dataset):

    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms is not None:
            self.transforms(data)

        return data, self.labels[index]


def normalize_dataset(dataset, mean=None, std=None):
    shape = dataset.shape
    channel_idx = np.where(np.array(shape) == 3)[0]

    # modify shape to [N, C, H, W]
    if channel_idx == 2:
        dataset = np.transpose(dataset, [0, 2, 1, 3])
    if channel_idx == 3:
        dataset = np.transpose(dataset, [0, 3, 1, 2])

    # [2024-10-10 Add by sunzekun]
    # 下面的代码会引发bug，因为目前数据集都是已经经过了归一化的
    # 此时有部分值会超出1，为1.xxx。但它不是像素值
    # 所以会错误地触发这个判断条件，导致整体所有的值再被除了一次255.
    # 为了避免出问题，此处直接把这行注释掉即可。

    # normalize
    # if (dataset[0] > 1).any():
    #     dataset = dataset / 255.

    # gaussian normalize
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        dataset = (dataset - mean) / std

    return dataset


def transform_data(data):
    shape = data.shape[1:]
    data_mod = random_crop(data, shape)
    data_mod = random_horiz_flip(data_mod)
    return data_mod


class MixupDataset(Dataset):
    def __init__(
        self,
        data_pair,
        label_pair,
        mixup_alpha=0.2,
        transforms=None,
        mean=None,
        std=None,
    ):
        # modify shape to [N, H, W, C]
        self.data_first = data_pair[0]
        self.data_second = data_pair[1]
        self.data_first = normalize_dataset(self.data_first, mean, std)
        self.data_second = normalize_dataset(self.data_second, mean, std)
        self.label_first = label_pair[0]
        self.label_second = label_pair[1]
        self.mixup_alpha = mixup_alpha
        self.transforms = transforms

    def __len__(self):
        return len(self.label_first)

    def __getitem__(self, index):
        lbd = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        if lbd < 0.5:
            lbd = 1 - lbd

        data_first = self.data_first[index]
        label_first = self.label_first[index]
        rnd_idx = np.random.randint(len(self.data_second))
        data_rnd_ax = self.data_second[rnd_idx]
        label_rnd_ax = self.label_second[rnd_idx]

        if self.transforms is not None:
            data_first = self.transforms(data_first)
            data_rnd_ax = self.transforms(data_rnd_ax)

        mixed_data = lbd * data_first + (1 - lbd) * data_rnd_ax
        mixed_labels = lbd * label_first + (1 - lbd) * label_rnd_ax
        return mixed_data, mixed_labels


class NormalizeDataset(BaseTensorDataset):
    def __init__(self, data, labels, transforms=None, mean=None, std=None):
        super().__init__(data, labels, transforms)
        self.data = normalize_dataset(data, mean, std)


def get_dataset_loader(
    dataset_name,
    loader_name,
    case,
    step=None,
    mean=None,
    std=None,
    batch_size=64,
    num_classes=0,
    drop_last=False,
    shuffle=False,
    onehot_enc=False,
    num_workers=0,
):
    """
    根据 loader_name 加载相应的数据集：支持增量训练 (inc)、辅助数据 (aux) 、测试数据 (test)和 D0数据集(train)
    """
    if not isinstance(loader_name, (list, tuple)):
        loader_name = [loader_name]

    data = []
    labels = []
    for ld_name in loader_name:
        data_path = settings.get_dataset_path(
            dataset_name, case, f"{ld_name}_data", step
        )
        label_path = settings.get_dataset_path(
            dataset_name, case, f"{ld_name}_label", step
        )

        print(f"Loading {data_path}")

        data.append(np.load(data_path))
        label = np.load(label_path)
        labels.append(label.astype(np.int64))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    transforms = None  # torchvision.transforms.Compose([])
    # if loader_name == "train":
    #     transform = True

    if onehot_enc:  # train label change to onehot for teacher model
        labels = np.eye(num_classes)[labels]

    # 构建自定义数据集
    dataset = NormalizeDataset(data, labels, transforms=transforms, mean=mean, std=std)
    # dataset = BaseTensorDataset(data, labels, transforms=transforms)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers
    )

    # if dataset_name == "pet-37":
    #     data_loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         drop_last=drop_last,
    #         shuffle=shuffle,
    #         num_workers=64,
    #         pin_memory=True
    #     )

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

    # # 测试加载增量数据集
    # print("Loading Incremental Training Dataset (inc)")
    # inc_dataset, inc_loader = get_dataset_loader("inc", data_dir, batch_size)
    # print(f"Incremental Dataset Size: {len(inc_dataset)}")
    #
    # # 遍历一批增量数据并查看形状
    # for images, labels in inc_loader:
    #     print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
    #     break  # 只打印第一批
    #
    # # 测试加载辅助数据集
    # print("\nLoading Auxiliary Dataset (aux)")
    # aux_dataset, aux_loader = get_dataset_loader("aux", data_dir, batch_size)
    # print(f"Auxiliary Dataset Size: {len(aux_dataset)}")
    #
    # # 遍历一批辅助数据并查看形状
    # for images, labels in aux_loader:
    #     print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
    #     break
    #
    # # 测试加载测试数据集
    # print("\nLoading Test Dataset (test)")
    # test_dataset, test_loader = get_dataset_loader("test", data_dir, batch_size)
    # print(f"Test Dataset Size: {len(test_dataset)}")
    #
    # # 遍历一批测试数据并查看形状
    # for images, labels in test_loader:
    #     print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
    #     break

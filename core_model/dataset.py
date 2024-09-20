from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, data, label):
        data = data.astype(np.float32)
        shape = data.shape
        channel_idx = np.where(np.array(shape) == 3)[0]
        if channel_idx == 2:
            data = np.transpose(data, [0, 2, 1, 3])
        if channel_idx == 3:
            data = np.transpose(data, [0, 3, 1, 2])

        # todo 保证所有数据集都正确
        if (data > 1).any():
            self.data = data / 255
        else:
            self.data = data

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_dataset_loader(
    loader_name, data_dir, batch_size, drop_last=False, shuffle=False
):
    """
    根据 loader_name 加载相应的数据集：支持增量训练 (inc)、辅助数据 (aux) 和测试数据 (test)。
    """
    # todo 待确定路径以及文件名称
    data_name, label_name = "", ""

    if loader_name == "inc":
        data_name, label_name = "cifar10_inc_data.npy", "cifar10_inc_labels.npy"
    elif loader_name == "aux":
        data_name, label_name = "cifar10_aux_data.npy", "cifar10_aux_labels.npy"
    elif loader_name == "test":
        data_name, label_name = "cifar10_test_data.npy", "cifar10_test_labels.npy"
    else:
        raise ValueError(
            f"Invalid loader_name {loader_name}. Choose from 'inc', 'aux', or 'test'."
        )

    data_path = os.path.join(data_dir, data_name)
    label_path = os.path.join(data_dir, label_name)

    # 检查文件是否存在
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"{data_name} or {label_name} not found in {data_dir}")

    data = np.load(data_path)
    label = np.load(label_path)

    # 构建自定义数据集
    dataset = CustomDataset(data, label)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle
    )

    return dataset, data_loader


if __name__ == "__main__":
    # 假设你的 CIFAR-10 数据存储在这个目录
    data_dir = "./data/cifar-10/noise/"
    # data_dir = "../data/cifar-100/noise/"
    # data_dir = "../data/tiny-imagenet-200/noise/"
    # data_dir = "../data/flowers-102/noise/"
    batch_size = 32

    # 测试加载增量数据集
    print("Loading Incremental Training Dataset (inc)")
    inc_dataset, inc_loader = get_dataset_loader("inc", data_dir, batch_size)
    print(f"Incremental Dataset Size: {len(inc_dataset)}")

    # 遍历一批增量数据并查看形状
    for images, labels in inc_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break  # 只打印第一批

    # 测试加载辅助数据集
    print("\nLoading Auxiliary Dataset (aux)")
    aux_dataset, aux_loader = get_dataset_loader("aux", data_dir, batch_size)
    print(f"Auxiliary Dataset Size: {len(aux_dataset)}")

    # 遍历一批辅助数据并查看形状
    for images, labels in aux_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break

    # 测试加载测试数据集
    print("\nLoading Test Dataset (test)")
    test_dataset, test_loader = get_dataset_loader("test", data_dir, batch_size)
    print(f"Test Dataset Size: {len(test_dataset)}")

    # 遍历一批测试数据并查看形状
    for images, labels in test_loader:
        print(f"Batch Image Shape: {images.shape}, Batch Label Shape: {labels.shape}")
        break

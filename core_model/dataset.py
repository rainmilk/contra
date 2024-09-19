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


def get_dataset_loader(loader_name, data_dir, batch_size, drop_last=False, shuffle=False):
    # todo 待确定路径以及文件名称
    data_name, label_name = '', ''
    if loader_name == 'inc':
        data_name, label_name = 'inc_data.npy', 'inc_label.npy'

    elif loader_name == 'aux':
        data_name, label_name = 'aux_data.npy', 'aux_label.npy'

    elif loader_name == 'test':
        data_name, label_name = 'test_data.npy', 'test_label.npy'

    data_path = os.path.join(data_dir, data_name)
    label_path = os.path.join(data_dir, label_name)
    data = np.load(data_path)
    label = np.load(label_path)

    dataset = CustomDataset(data, label)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)

    return dataset, data_loader

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from core_model.custom_model import ClassifierWrapper, load_custom_model
from configs import settings
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTensorDataset(Dataset):
    def __init__(self, data, labels, transforms=None, device=None):
        self.data = torch.as_tensor(data, device=device)
        self.labels = torch.as_tensor(labels, device=device)
        self.transforms = transforms
        logger.info(f"Dataset initialized with data shape: {data.shape} and labels shape: {labels.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms is not None:
            self.transforms(data)

        return data, self.labels[index]
    
def get_model_path(dataset_name, model_name, method):
    
    # TODO-241105 [sunzekun] 新的参数读取路径规则，先做初步测试用，对各个数据集需要分别根据文档的设定手写case（主要变的是noise_ratio)
    base_ckpt_path = f"/nvme/szh/code/tta-mr/ckpt/{dataset_name}/nr_0.25_nt_asymmetric_cvpr"
    if method == 'pretrain':
        return f"/nvme/szh/code/tta-mr/ckpt/{dataset_name}/pretrain/{model_name}_pretrain.pth"
    elif method == 'inc_train':
        return os.path.join(base_ckpt_path, f"{model_name}_inc_train.pth")
    else:
        return os.path.join(base_ckpt_path, method, f"{model_name}_restore.pth")
    
    
def load_model_and_data(
    dataset_name, model_name, method
):
    """
    加载模型及其对应的测试数据集，并返回模型和数据加载器
    """
    # 更改数据集为 pet-37
    # dataset_name = "pet-37"  # 使用 Oxford-IIIT Pet 数据集
    

    num_classes = settings.num_classes_dict[dataset_name]

    # 加载测试数据集
    # Load test dataset
    batch_size = 64
    test_data_path = f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/test_data.npy"
    test_label_path = f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/test_label.npy"

    try:
        logger.info(f"Loading test data from: {test_data_path}")
        test_data = np.load(test_data_path)
        test_labels = np.load(test_label_path)
        logger.info(f"Test data loaded with shape: {test_data.shape}, Test labels loaded with shape: {test_labels.shape}")
        test_dataset = BaseTensorDataset(test_data, test_labels, device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except FileNotFoundError as e:
        logger.error(f"Error loading test dataset: {e}")
        return


    # 加载模型    
    num_classes = settings.num_classes_dict[dataset_name]
    logger.info(f"Currently testing dataset: {dataset_name} with {num_classes} classes.")

    model_path = get_model_path(dataset_name, model_name, method)
    logger.info(f"Evaluating model from path: {model_path}")

    # Load model architecture and weights
    model = load_custom_model(model_name, num_classes, load_pretrained=False)
    model = ClassifierWrapper(model, num_classes)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Loaded model checkpoint from {model_path}")
    except FileNotFoundError:
        logger.error(f"Cannot find the weight file at {model_path}. Skipping.")
        return
    model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.to(device)

    return model, test_dataloader, test_labels

# 获取嵌入数据
def get_embeddings(model, dataloader, layer_name="feature_model"):
    """
    提取模型的嵌入层输出
    """
    embeddings = []
    labels = []

    # 注册hook来捕获指定层的输出
    def hook_fn(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = model._modules.get(layer_name).register_forward_hook(hook_fn)

    # 进行前向传播以获取嵌入
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            labels.extend(targets.cpu().numpy())

    handle.remove()

    # 拼接所有批次的嵌入和标签
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    return embeddings, labels

# 绘制 t-SNE
def get_tsne(model, dataloader, class_range=None):
    """
    绘制多个step的t-SNE结果，展示不同方法在不同步骤的嵌入变化
    """
    embeddings, labels = get_embeddings(model, dataloader)

    return labels

def plot_tsne(
    methods, dataset_name, model_name, noise_type, class_range=None, title=None, use_cache=False
):
    for method in methods:
        model = None
        dataloader = None
        if model is None or dataloader is None:
            model, dataloader, _ = load_model_and_data(
                dataset_name, model_name, method
            )
        labels = get_tsne(
            model, dataloader
        )


# 执行 t-SNE 可视化

methods = [ # baseline方法，包括一些预训练，增量训练(raw, 仅在d1上训练) 和基础的baseline
            # 'pretrain', 'inc_train', 'finetune',
            'pretrain', 'inc_train',
            # LNL方法
            # 'Coteaching', 'Coteachingplus', 'JoCoR', 'Decoupling', 'NegativeLearning', 'PENCIL',                
            'Coteaching',
            # MU 方法
            # 'raw', 'GA', 'GA_l1', 'FT', 'FT_l1', 'fisher_new', 'wfisher', 'FT_prune', 'FT_prune_bi', 'retrain', 'retrain_ls', 'retrain_sam',
            'wfisher',
            # 我们的方法
            'CRUL'
            ]
dataset_name = "pet-37"  # 更改为 pet-37 数据集
noise_type= "asymmetric"
model_name = "wideresnet50"  # 确保模型适配 pet-37 数据集

class_range = [1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 28, 29, 30, 31, 34, 35, 36] # 狗的类别

title = None

plot_tsne(
    methods, dataset_name, model_name, noise_type, class_range=class_range, title=title, use_cache=False
)

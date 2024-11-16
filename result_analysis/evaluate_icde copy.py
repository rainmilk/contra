import warnings
warnings.filterwarnings("ignore")
import argparse
from args_paser import parse_args
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
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.train_test import model_forward
from sklearn.metrics.pairwise import cosine_similarity


os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTensorDataset(Dataset):
    def __init__(self, data, labels, transforms=None, device=None):
        self.data = torch.as_tensor(data, device=device)
        self.labels = torch.as_tensor(labels, device=device)
        self.transforms = transforms
        print(f"Dataset initialized with data shape: {data.shape} and labels shape: {labels.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transforms is not None:
            self.transforms(data)

        return data, self.labels[index]
    
def load_model_and_data(args,dataset_name, model_name, method):
    """
    加载模型及其对应的测试数据集，并返回模型和数据加载器
    """

    num_classes = settings.num_classes_dict[dataset_name]

    # 加载测试数据集
    # Load test dataset
    batch_size = 64
    case = settings.get_case(args.noise_ratio, args.noise_type)
    
    test_data_path = f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/test_data.npy"
    test_label_path = f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/test_label.npy"
    
    target_data_path=f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/nr_0.5_nt_symmetric_cvpr/train_noisy_data.npy"
    target_label_path=f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/nr_0.5_nt_symmetric_cvpr/train_noisy_label.npy"
    target_true_label_path=f"/nvme/szh/code/tta-mr/data/{dataset_name}/gen/nr_0.5_nt_symmetric_cvpr/train_noisy_true_label.npy"

    try:
        print(f"Loading test data from: {test_data_path}")
        test_data = np.load(test_data_path)
        test_labels = np.load(test_label_path)
        
        target_data = np.load(target_data_path)
        target_labels = np.load(target_label_path)
        target_true_labels = np.load(target_true_label_path)
        
        print(f"Test data loaded with shape: {test_data.shape}, Test labels loaded with shape: {test_labels.shape}")
        
        test_dataset = BaseTensorDataset(test_data, test_labels, device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        target_dataset = BaseTensorDataset(target_data, target_labels, device=device)
        target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    except FileNotFoundError as e:
        print(f"Error loading test dataset: {e}")
        return


    # 加载模型    
    num_classes = settings.num_classes_dict[dataset_name]
    print(f"Currently testing dataset: {dataset_name} with {num_classes} classes.")

    # Load model architecture and weights
    model = load_custom_model(model_name, num_classes, load_pretrained=False)
    model = ClassifierWrapper(model, num_classes)
    
    model_path="ckpt/cifar-10/nr_0.25_nt_symmetric_cvpr/Coteaching/cifar-resnet18_restore.pth"
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Loaded model checkpoint from {model_path}")
    except FileNotFoundError:
        print(f"Cannot find the weight file at {model_path}. Skipping.")
        return
    model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.to(device)

    return model, target_dataloader,test_dataloader, target_labels,target_true_labels,test_labels

# 获取嵌入数据
def get_embeddings(model, dataloader, layer_name="feature_model"):
    """
    提取模型的嵌入层输出
    """
    embeddings = []
    #labels = []

    # 注册hook来捕获指定层的输出
    def hook_fn(module, input, output):
        embeddings.append(output.detach().cpu().numpy())

    handle = model._modules.get(layer_name).register_forward_hook(hook_fn)

    # 进行前向传播以获取嵌入
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            #labels.extend(targets.cpu().numpy())

    handle.remove()

    # 拼接所有批次的嵌入和标签
    embeddings = np.concatenate(embeddings, axis=0)
    #labels = np.array(labels)
    return embeddings
    #return embeddings, labels


def search(target_embeddings,query_embeddings,target_true_labels,target_labels,test_labels,k=5):
    results=[]
    i=0
    for query in query_embeddings:
        similarities=cosine_similarity([query],target_embeddings)[0]
        top_k_indices=np.argsort(similarities)[-k:][::-1]
        top_k_similarities=[(index,similarities[index],target_labels[index],target_true_labels[index])for index in top_k_indices]
        top_k_similarities.insert(0,test_labels[i])
        results.append(top_k_similarities)
        i=i+1
    # print(results[0]) 
    print(len(results))#10000
    print(len(results[0]))
    return results
    
def get_data_embeddings(methods, dataset_name, model_name, noise_type):
    for method in methods:
        model = None
        dataloader = None
        if model is None or dataloader is None:
            model, target_dataloader,test_dataloader,target_labels,target_true_labels,test_labels = load_model_and_data(
                dataset_name, model_name, method
            )
        query_embeddings= get_embeddings(
            model, test_dataloader
        )
        target_embeddings = get_embeddings(
            model, target_dataloader
        )
        return query_embeddings,target_embeddings,target_labels,target_true_labels,test_labels
    

methods = [ # baseline方法，包括一些预训练，增量训练(raw, 仅在d1上训练) 和基础的baseline
            # 'pretrain', 'inc_train', 'finetune',
            #'pretrain', 'inc_train',
            # LNL方法
            # 'Coteaching', 'Coteachingplus', 'JoCoR', 'Decoupling', 'NegativeLearning', 'PENCIL',                
            'Coteaching',
            # MU 方法
            # 'raw', 'GA', 'GA_l1', 'FT', 'FT_l1', 'fisher_new', 'wfisher', 'FT_prune', 'FT_prune_bi', 'retrain', 'retrain_ls', 'retrain_sam',
            #'wfisher',
            # 我们的方法
            #'CRUL'
            ]

# dataset_name = "cifar-10"
# noise_type= "symmetric"
# model_name = "cifar-resnet18"  

# k=5

# query_embeddings,target_embeddings,target_labels,target_true_labels,test_labels = get_data_embeddings(methods, dataset_name, model_name, noise_type)

# results=search(target_embeddings,query_embeddings,target_labels,target_true_labels,test_labels,k)
# print(results[0])

if __name__ == "__main__":
    try:
        args = parse_args()
        case = settings.get_case(args.noise_ratio, args.noise_type)
        uni_names = args.uni_name
        uni_names = [uni_names] if uni_names is None else uni_names.split(",")
        num_classes = settings.num_classes_dict[args.dataset]
        #加载model
        #加载dataset
        test_data, test_labels, test_loader = get_dataset_loader(
            args.dataset,
            "test",
            None,
            batch_size=args.batch_size,
            shuffle=False,
        )
        
        target_data, target_true_labels, target_loader = get_dataset_loader(
        args.dataset,
        "train_noisy",
        case,
        batch_size=args.batch_size,
        shuffle=False,
        label_name="train_noisy_true_label"
        )
        _,target_labels,_ = get_dataset_loader(
        args.dataset,
        "train_noisy",
        case,
        batch_size=args.batch_size,
        shuffle=False,
        label_name="train_noisy_label"
        )
        
        loaded_model = load_custom_model(args.model, num_classes, load_pretrained=False)
        model = ClassifierWrapper(loaded_model, num_classes)
        
        model_ckpt_path = settings.get_ckpt_path(
            args.dataset,
            case,
            args.model,
            model_suffix=args.model_suffix,
            unique_name=uni_names[0],
        )
        
        print(f"Loading model from {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        
        _ ,_ ,query_embeddings,_ = model_forward(
        test_loader, model, device, output_embedding=True, output_targets=True
        )
        _ ,_ ,target_embeddings,_ = model_forward(
        target_loader, model, device, output_embedding=True, output_targets=True
        )
        k=5
        results=search(target_embeddings,query_embeddings,target_true_labels,target_labels,test_labels,k)
        print(results[0])
        
    except argparse.ArgumentError as e:
            print(f"Error parsing arguments: {e}")

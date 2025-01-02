import warnings
warnings.filterwarnings("ignore")
import argparse
from args_parser import parse_args
import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from configs import settings
import logging
from configs import settings
from core_model.custom_model import ClassifierWrapper, load_custom_model
from core_model.dataset import get_dataset_loader
from core_model.train_test import model_forward
from core_model.dataset import BaseTensorDataset
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(file_path, is_data=True):
    """
    加载数据集文件并返回 PyTorch 张量。
    :param subdir: 数据目录
    :param dataset_name: 数据集名称 (cifar-10, cifar-100, food-101, pet-37, flower-102)
    :param file_name: 数据文件名
    :param is_data: 是否为数据文件（True 表示数据文件，False 表示标签文件）
    :return: PyTorch 张量格式的数据
    """
    data = np.load(file_path)

    if is_data:
        # 对于数据文件，转换为 float32 类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
    else:
        # 对于标签文件，转换为 long 类型
        data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor

# def search(target_embeddings,query_embeddings,target_true_labels,target_labels,test_labels,k=5):
#     results=[]
#     i=0
#     for query in query_embeddings:
#         similarities=cosine_similarity([query],target_embeddings)[0]
#         top_k_indices=np.argsort(similarities)[-k:][::-1]
#         top_k_similarities=[(index,similarities[index],target_labels[index],target_true_labels[index])for index in top_k_indices]
#         top_k_similarities.insert(0,test_labels[i])
#         results.append(top_k_similarities)
#         i=i+1
#     # print(results[0]) 
#     print(len(results))#10000
#     print(len(results[0]))
#     print("search finish")
#     return results

def search(target_embeddings, query_embeddings, target_true_labels, target_labels, test_labels, k=5):
    results = []
    target_embeddings = torch.tensor(target_embeddings,dtype=torch.float64).to(device)
    query_embeddings = torch.tensor(query_embeddings,dtype=torch.float64).to(device)

    for i, query in enumerate(query_embeddings):
        distances = torch.sqrt(torch.sum((target_embeddings - query) ** 2, dim=1)).cpu().numpy()

        top_k_indices = np.argsort(distances)[:k]
        top_k_distances = [(index, distances[index], target_labels[index], target_true_labels[index]) for index in top_k_indices]
        
        top_k_distances.insert(0, test_labels[i])
        results.append(top_k_distances)

    print(len(results)) # 10000
    print(len(results[0]))
    print("search finish")
    return results


def recall_k(results,target_true_labels,counter,k=5):
    total_recall=0
    for each in results:
        query_label= each[0]
        target_items=each[1:-1]
        temp_recall=0
        top_k_items=target_items[:k]
        #目标集中实际正类数量
        sum_target_true=counter[query_label]
        #前k个预测中正确的正类数量
        sum_predict_true=sum(1 for _,_,_,true_label in top_k_items if true_label==query_label)
        temp_recall=sum_predict_true/sum_target_true
        total_recall=total_recall+temp_recall
        
    return total_recall/len(results)

def precision_k(results,k):
    total_precision=0
    for each in results:
        query_label= each[0]
        target_items=each[1:-1]
        temp_precision=0
        top_k_items=target_items[:k]
        sum_true=k
        #前k个预测中正确的正类数量
        sum_predict_true=sum(1 for _,_,_,true_label in top_k_items if true_label==query_label)
        temp_precision=sum_predict_true/sum_true
        total_precision=total_precision+temp_precision
        
    return total_precision/len(results)

# def map_k(results,target_true_labels,k=50):
#     total_recall=0
#     for each in results:
#         query_label= each[0]
#         target_items=each[1:-1]
#         temp_recall=0
#         top_k_items=target_items[:k]
#         #目标集中实际正类数量
#         sum_target_true=sum(1 for label in target_true_labels if label==query_label)
#         #前k个预测中正确的正类数量
#         sum_predict_true=sum(1 for _,_,_,true_label in top_k_items if true_label==query_label)
#         temp_recall=sum_predict_true/sum_target_true
#         total_recall=total_recall+temp_recall
        
#     return total_recall/len(results)
        
def get_metrics(results,target_true_labels):
    #results[0]
    # [3, (1639, 0.9648188, 1, 2), (2193, 0.95367, 0, 3), (2409, 0.94515646, 9, 3), (2195, 0.941772, 3, 3), (2187, 0.934531, 9, 3)]
    #[query的class,(targetset的index，相似度，noisy_label，true_label),...]
    pass    
def execute(args):
    case = settings.get_case(args.noise_ratio, args.noise_type)
    uni_names = args.uni_name
    uni_names = [uni_names] if uni_names is None else uni_names.split(",")
    num_classes = settings.num_classes_dict[args.dataset]

    test_data, test_labels, test_loader = get_dataset_loader(
        args.dataset,
        "test",
        None,
        batch_size=args.batch_size,
        shuffle=False,
    )
    #得到target_data,target_true_labels,target_labels,target_loader

    dataset_name=args.dataset
    batch_size=args.batch_size
    
    all_data=False
    if all_data:
        train_clean_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "train_clean_data")
        )
        
        train_clean_labels=(np.load(settings.get_dataset_path(dataset_name, case, "train_clean_label"))).astype(np.int64)
        
        train_noisy_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "train_noisy_data")
        )
        
        train_noisy_true_labels=(np.load(settings.get_dataset_path(dataset_name, case, "train_noisy_true_label"))).astype(np.int64)
        train_noisy_labels = (np.load(settings.get_dataset_path(dataset_name, case, "train_noisy_label"))).astype(np.int64)
        
        train_data = torch.concatenate([train_clean_data, train_noisy_data])
        train_labels = np.concatenate([train_clean_labels, train_noisy_labels])
        
        target_true_labels=np.concatenate([train_clean_labels, train_noisy_true_labels])
        target_labels=np.concatenate([train_clean_labels, train_noisy_labels])
        
        target_data = BaseTensorDataset(train_data, train_labels)
        target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False)
    else:
        train_noisy_data = load_dataset(
            settings.get_dataset_path(dataset_name, case, "train_noisy_data")
        )
        
        train_noisy_true_labels=(np.load(settings.get_dataset_path(dataset_name, case, "train_noisy_true_label"))).astype(np.int64)
        train_noisy_labels = (np.load(settings.get_dataset_path(dataset_name, case, "train_noisy_label"))).astype(np.int64)
        
        target_true_labels= train_noisy_true_labels
        target_labels= train_noisy_labels
        
        target_data = BaseTensorDataset(train_noisy_data, train_noisy_labels)
        target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False)
    
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
    
    from collections import Counter
    counter=Counter(target_true_labels)

    result_nums=10 #每个查询返回几个结果
    
    results=search(target_embeddings,query_embeddings,target_true_labels,target_labels,test_labels,result_nums)
    print(results[0]) 
    print(len(results))
     #[query的label,(targetset的index，相似度，noisy_label，true_label),...]
    print(recall_k(results,target_true_labels,counter))
        
if __name__ == "__main__":
    try:
        pargs = parse_args()
        execute(pargs)
    except argparse.ArgumentError as e:
            print(f"Error parsing arguments: {e}")

# methods = [ # baseline方法，包括一些预训练，增量训练(raw, 仅在d1上训练) 和基础的baseline
#             # 'pretrain', 'inc_train', 'finetune',
#             #'pretrain', 'inc_train',
#             # LNL方法
#             # 'Coteaching', 'Coteachingplus', 'JoCoR', 'Decoupling', 'NegativeLearning', 'PENCIL',                
#             'Coteaching',
#             # MU 方法
#             # 'raw', 'GA', 'GA_l1', 'FT', 'FT_l1', 'fisher_new', 'wfisher', 'FT_prune', 'FT_prune_bi', 'retrain', 'retrain_ls', 'retrain_sam',
#             #'wfisher',
#             # 我们的方法
#             #'CRUL'
#             ]
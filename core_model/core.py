import os
import logging

import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np

from main import parse_args, parse_kwargs
from nets.VGG_LTH import vgg16_bn_lth
from lip_teacher import SimpleLipNet
from dataset import CustomDataset, get_dataset_loader
from train_test import (
    model_train,
    model_test,
    working_model_forward,
    teacher_model_forward,
)


# todo 数据集路径和类别数量
dataset_paths = {
    "cifar-10": "../data/cifar-10",
    "cifar-100": "../data/cifar-100",
    "flowers-102": "../data/flowers-102",
    "tiny-imagenet-200": "../data/tiny-imagenet-200",
}


num_classes_dict = {
    "cifar-10": 10,
    "cifar-100": 100,
    "flowers-102": 102,
    "tiny-imagenet-200": 200,
}


def iterate_repair_model(
    working_model,
    working_opt,
    working_criterion,
    working_model_save_path,
    teacher_model,
    teacher_opt,
    teacher_criterion,
    teacher_model_save_path,
    alpha,
    inc_dataset,
    inc_dataloader,
    aux_dataset,
    aux_dataloader,
    num_classes,
    args,
):
    # 1. 通过 Mt 获取 D_mix=Da+Ds+Dc,  Train Pp=Mp(X_mix), Loss=CrossEntropy(Pp, Y_mix)
    # (1) 获取Ds: 通过 Yp=Mt(Xtr) 预测分类标签，得到Ds, 其中Yp=Ytr
    inc_embeddings, inc_predicts = teacher_model_forward(teacher_model, inc_dataloader)
    inc_data, inc_labels = inc_dataset.data, inc_dataset.label
    selected_data, selected_labels = inc_dataset[inc_predicts == inc_labels]

    # (2) 获取Dc: 通过 Mt(Xa) 计算class embedding centroids (i.e. Class mean): E_centroid
    # 通过 Mt(Xtr) 获取 embed_tr, 提取离每个类c的中心E_centroid[class=c]最近的Top 20%的数据为 Dc (通过Lipschitz性质预测的伪标签)
    aux_embeddings, aux_predicts = teacher_model_forward(teacher_model, aux_dataloader)
    aux_data, aux_labels = aux_dataset.data, aux_dataset.label
    centroid_data, centroid_labels = [], []

    for label in list(set(aux_labels)):
        aux_class_embedding = aux_embeddings[aux_labels == label]
        auc_class_embedding_centroid = np.mean(aux_class_embedding, axis=0)

        inc_class_embeddings = inc_embeddings[inc_labels == label]
        inc_class_data, inc_class_label = inc_dataset[inc_labels == label]
        distances = np.linalg.norm(
            inc_class_embeddings - auc_class_embedding_centroid, axis=-1
        )
        selected_topk_num = int(len(inc_class_label) * 0.2)
        top_idx = np.argpartition(distances, -selected_topk_num)[-selected_topk_num:]

        centroid_class_data, centroid_class_label = (
            inc_class_data[top_idx],
            inc_class_label[top_idx],
        )
        centroid_data.extend(centroid_class_data)
        centroid_labels.extend(centroid_class_label)

    centroid_data = np.array(centroid_data)

    # (3) train Mp: Train Pp=Mp(X_mix), Loss=CrossEntropy(Pp, Y_mix)
    mix_data = np.concatenate([aux_data, selected_data, centroid_data], axis=0)
    mix_labels = np.concatenate([aux_labels, selected_labels, centroid_labels], axis=0)
    mix_labels_onehot = np.eye(num_classes)[mix_labels]
    mix_dataset = CustomDataset(mix_data, mix_labels_onehot)
    mix_dataloader_shuffled = DataLoader(
        mix_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True
    )

    model_train(
        mix_dataloader_shuffled,
        working_model,
        working_opt,
        working_criterion,
        alpha,
        args,
        save_path=working_model_save_path,
    )

    # 2. 通过 Mp 获取 D_conf={X_conf, P_conf}, Train Pt=Mt(Xconf), Loss=CrossEntropy(Pt, Pconf)
    # (1) 获取Dconf: 通过 Ptr=Mp(Xtr)，根据 confidence(N*C概率矩阵)，sample每个类20%数据Dconf={Xconf, Pconf}
    inc_probs, inc_predicts = working_model_forward(inc_dataloader, working_model)
    conf_topk_num = 100
    top_idx = []

    for label in list(set(inc_labels)):
        inc_class_probs = inc_probs[:, label]
        top_class_idx = np.argpartition(inc_class_probs, -conf_topk_num)[
            -conf_topk_num:
        ]

        top_idx.extend(top_class_idx)

    conf_data, conf_probs = inc_data[top_idx], inc_probs[top_idx]
    conf_dataset = CustomDataset(conf_data, conf_probs)
    conf_dataloader_shuffled = DataLoader(
        conf_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True
    )

    # (2) train Mt:  Pt=Mt(Xconf)，Loss=CrossEntropy(Pt, Pconf)
    model_train(
        conf_dataloader_shuffled,
        teacher_model,
        teacher_opt,
        teacher_criterion,
        alpha,
        args,
        teacher_model_save_path,
        teacher_model=True,
    )
    return conf_dataset


def iterate_adapt_model(
    working_model,
    working_opt,
    working_criterion,
    working_model_save_path,
    teacher_model,
    teacher_opt,
    teacher_criterion,
    teacher_model_save_path,
    alpha,
    aug_data,
    aug_probs,
    inc_dataset,
    inc_dataloader,
    args,
):
    # 1. 构造Dts融合数据集 Dp_mix: (Dts, D_aug), 进行mix up
    # (1) 构造 Dp: Dp={Xts, Pp}, Pp = Mp(Xts)
    inc_data, inc_labels = inc_dataset.data, inc_dataset.label
    inc_probs, inc_predicts = working_model_forward(inc_dataloader, working_model)

    # (2) 构造 Dp_mix: Dp_mix = mix_up(Dp, D_aug), Xp_mix = {a*Xp+(1-a)*X_aug}, Yp_mix = {a*Yp+(1-a)*Y_aug}
    ts_mixed_dataloader_shuffled = get_ts_mixed_data(
        aug_data, aug_probs, inc_data, inc_probs, args.batch_size
    )

    # 2. train Mt: Pt=Mt(Xp_max), Update Mt: Loss=CrossEntropy(Pt, Yp_mix)
    model_train(
        ts_mixed_dataloader_shuffled,
        teacher_model,
        teacher_opt,
        teacher_criterion,
        alpha,
        args,
        save_path=teacher_model_save_path,
        teacher_model=True,
    )

    # 3. 重新构造 Dts融合数据集 Dp_mix
    # (1) 构造 Dp: Dp={Xts, Pp}, Pp = Mp(Xts)
    inc_probs_new, inc_predicts_new = working_model_forward(
        inc_dataloader, working_model
    )
    # (2) 构造 Dp_mix: Dp_mix = mix_up(Dp, D_aug), Xp_mix = {a*Xp+(1-a)*X_aug}, Yp_mix = {a*Yp+(1-a)*Y_aug}
    ts_mixed_dataloader_shuffled_new = get_ts_mixed_data(
        aug_data, aug_probs, inc_data, inc_probs_new, args.batch_size
    )

    # 4. train Mp: Pp=Mp(Xp_max), Update Mp: Loss=CrossEntropy(Pp, Yp_mix)
    model_train(
        ts_mixed_dataloader_shuffled_new,
        working_model,
        working_opt,
        working_criterion,
        alpha,
        args,
        save_path=working_model_save_path,
    )


def get_ts_mixed_data(aug_data, aug_probs, inc_data, inc_probs, batch_size, alpha=0.75):
    aug_size, inc_size = len(aug_probs), len(inc_data)
    sampling_random_idx = np.random.choice(np.arange(aug_size), inc_size)
    aug_data_sampling = aug_data[sampling_random_idx]
    aug_probs_sampling = aug_probs[sampling_random_idx]

    lambda_from_beta = np.random.beta(alpha, alpha, size=inc_size)

    compare_data = np.concatenate(
        [lambda_from_beta[:, np.newaxis], (1 - lambda_from_beta)[:, np.newaxis]],
        axis=-1,
    )
    lambda_from_beta = np.max(compare_data, axis=-1)
    lambda_for_data = lambda_from_beta[:, np.newaxis, np.newaxis, np.newaxis]
    lambda_for_probs = lambda_from_beta[:, np.newaxis]

    ts_mixed_data = (
        lambda_for_data * inc_data + (1 - lambda_for_data) * aug_data_sampling
    )
    ts_mixed_probs = (
        lambda_for_probs * inc_probs + (1 - lambda_for_probs) * aug_probs_sampling
    )
    ts_mixed_dataset = CustomDataset(ts_mixed_data, ts_mixed_probs)
    ts_mixed_dataloader_shuffled = DataLoader(
        ts_mixed_dataset, batch_size, drop_last=True, shuffle=True
    )
    return ts_mixed_dataloader_shuffled


def get_model_paths(args, dataset):
    """Generate and return model paths dynamically."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(curr_dir)
    ckpt_dir = os.path.join(par_dir, "ckpt", dataset)

    return {
        "working_model_path": os.path.join(ckpt_dir, "pretrain_checkpoint.pth.tar"),
        "working_model_repair_save_path": os.path.join(
            ckpt_dir, "working_model_repair"
        ),
        "working_model_adapt_save_path": os.path.join(ckpt_dir, "working_model_adapt"),
        "lip_teacher_model_path": os.path.join(ckpt_dir, "lip_checkpoint.pth.tar"),
        "teacher_model_repair_save_path": os.path.join(
            ckpt_dir, "teacher_model_repair"
        ),
        "teacher_model_adapt_save_path": os.path.join(ckpt_dir, "teacher_model_adapt"),
    }


def execute(args):
    # 1. 获取公共参数
    num_classes = num_classes_dict[args.dataset]
    kwargs = parse_kwargs(args.kwargs)
    alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 0.1)

    # todo 可加入参数中
    # learning_rate = 0.001
    # weight_decay = 1e-4
    # repair_iter_num = 2
    # adapt_iter_num = 2

    learning_rate = getattr(args, "learning_rate", 0.001)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    repair_iter_num = getattr(args, "repair_iter_num", 2)
    adapt_iter_num = getattr(args, "adapt_iter_num", 2)

    model_paths = get_model_paths(args, args.dataset)

    working_model_path = model_paths["working_model_path"]
    working_model_repair_save_path = model_paths["working_model_repair_save_path"]
    working_model_adapt_save_path = model_paths["working_model_adapt_save_path"]
    lip_teacher_model_path = model_paths["lip_teacher_model_path"]
    teacher_model_repair_save_path = model_paths["teacher_model_repair_save_path"]
    teacher_model_adapt_save_path = model_paths["teacher_model_adapt_save_path"]

    working_model, lip_teacher_model = None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. load model
    # (1) load working model
    if args.model == "resnet18":
        working_model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif args.model == "vgg16":
        working_model = vgg16_bn_lth(num_classes=num_classes)

    working_model.to(device)

    working_opt = optim.Adam(
        working_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    working_criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(working_model_path)
    working_model.load_state_dict(checkpoint["state_dict"], strict=False)
    working_model.cuda()

    # (2) load lip_teacher model
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    lip_teacher_model = SimpleLipNet(resnet, 512, num_classes)
    teacher_opt = optim.Adam(
        lip_teacher_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    teacher_criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(lip_teacher_model_path)
    lip_teacher_model.load_state_dict(checkpoint["state_dict"], strict=False)
    lip_teacher_model.cuda()

    # 3. 迭代修复过程
    # (1) 构造修复过程数据集: Dtr、 Da、Dts
    data_dir = dataset_paths[args.dataset]
    inc_dataset, inc_dataloader = get_dataset_loader(
        "inc", data_dir, args.batch_size, shuffle=False
    )
    aux_dataset, aux_dataloader = get_dataset_loader(
        "aux", data_dir, args.batch_size, shuffle=False
    )
    test_dataset, test_dataloader = get_dataset_loader(
        "test", data_dir, args.batch_size, shuffle=False
    )

    # (2) 测试修复前 Dts 在 Mp 的表现
    test_result_before = model_test(test_dataset, test_dataloader, working_model)

    # (3) 迭代修复过程：根据 Dtr 迭代 Mp 、 Mt
    conf_dataset = None
    for i in range(repair_iter_num):
        conf_dataset = iterate_repair_model(
            working_model,
            working_opt,
            working_criterion,
            working_model_repair_save_path,
            lip_teacher_model,
            teacher_opt,
            teacher_criterion,
            teacher_model_repair_save_path,
            alpha,
            inc_dataset,
            inc_dataloader,
            aux_dataset,
            aux_dataloader,
            num_classes,
            args,
        )

    # 4. 测试修复后 Dts 在 Mp 的表现
    test_result_after_repair = model_test(test_dataset, test_dataloader, working_model)

    # 5. 迭代测试数据适应过程
    # (1) 构造适应过程数据：Dts, D_aug:  = Da + Dconf
    aux_data, aux_labels = aux_dataset.data, aux_dataset.label
    aux_labels_onehot = np.eye(num_classes)[aux_labels]
    conf_data, conf_probs = conf_dataset.data, conf_dataset.label

    aug_data = np.concatenate([aux_data, conf_data], axis=0)
    aug_probs = np.concatenate([aux_labels_onehot, conf_probs], axis=0)

    # (2) 迭代测试数据适应过程：根据 混合的Dts 迭代 Mp 和 Mt
    for i in range(adapt_iter_num):
        iterate_adapt_model(
            working_model,
            working_opt,
            working_criterion,
            working_model_adapt_save_path,
            lip_teacher_model,
            teacher_opt,
            teacher_criterion,
            teacher_model_adapt_save_path,
            alpha,
            aug_data,
            aug_probs,
            test_dataset,
            test_dataloader,
            args,
        )

    # 6. 测试适应后 Dts 在 Mp 的表现
    test_result_after_adapt = model_test(test_dataset, test_dataloader, working_model)

    print("---------------------test before------------------------------")
    print(test_result_before)
    print("---------------------test after repair------------------------------")
    print(test_result_after_repair)
    print("---------------------test after adapt-------------------------------")
    print(test_result_after_adapt)

    logging.basicConfig(filename="../logs/core_execution.log", level=logging.INFO)
    logging.info(f"Test results before repair: {test_result_before}")
    logging.info(f"Test results after repair: {test_result_after_repair}")
    logging.info(f"Test results after adaptation: {test_result_after_adapt}")


if __name__ == "__main__":
    # parse_args = parse_args()
    try:
        parse_args = parse_args()
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    execute(parse_args)

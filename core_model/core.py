import os
import logging
import argparse

import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import sys

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
    device,
    args,
):
    # 1. 通过 Mt 获取 Dp_mix=Da+Ds+Dc,  Train Pp=Mp(Xp_mix), Loss=CrossEntropy(Pp, Yp_mix)
    # (1) 获取Ds: 通过 Yp=Mp(Xtr), Yt=Mt(Xtr) 两个模型预测分类标签，其中Yp=Yt的数据为Ds, Ys为预测相同的标签
    working_inc_predicts, working_inc_probs = working_model_forward(inc_dataloader, working_model)
    teacher_inc_predicts, teacher_inc_probs, teacher_inc_embeddings = teacher_model_forward(inc_dataloader, teacher_model)

    agree_idx = working_inc_predicts == teacher_inc_predicts
    selected_data = inc_dataset.data[agree_idx]
    selected_labels = teacher_inc_predicts[agree_idx]
    selected_embeddings = teacher_inc_embeddings[agree_idx]

    # (2) 获取Dc: 通过 Mt(Xa+Xs) 计算class embedding centroids (i.e. Class mean): E_centroid
    # 通过 Mt(D_agree)获取 E_centroid(D_agree=Da+Ds)
    # 获取 Embedding_disa(D_disa=Dtr-D_agree) 距离离每个类c的中心 E_centroid[class=c]最近的Top 10%的数据为 Dc (通过Lipschitz性质预测的伪标签)
    aux_predicts, aux_probs, aux_embeddings = teacher_model_forward(aux_dataloader, teacher_model)
    aux_data, aux_labels = aux_dataset.data, aux_dataset.label

    teacher_agree_embeddings = np.concatenate([aux_embeddings, selected_embeddings], axis=0)
    agree_labels = np.concatenate([aux_labels, selected_labels], axis=0)

    disagree_idx = working_inc_predicts != teacher_inc_predicts
    disagree_data = inc_dataset.data[disagree_idx]
    teacher_disagree_predicts = teacher_inc_predicts[disagree_idx]
    teacher_disagree_probs = teacher_inc_probs[disagree_idx]
    teacher_disagree_embeddings = teacher_inc_embeddings[disagree_idx]

    centroid_data, centroid_labels = [], []

    for label in list(set(aux_labels)):
        agree_class_embedding = teacher_agree_embeddings[agree_labels == label]
        agree_class_embedding_centroid = np.mean(agree_class_embedding, axis=0)

        disagree_class_idx = teacher_disagree_predicts == label
        disagree_class_embeddings = teacher_disagree_embeddings[disagree_class_idx]
        disagree_class_data = disagree_data[disagree_class_idx]
        disagree_class_label = teacher_disagree_predicts[disagree_class_idx]

        distances = np.linalg.norm(
            disagree_class_embeddings - agree_class_embedding_centroid, axis=-1
        )
        selected_topk_num = len(disagree_class_label) // 10
        top_idx = np.argpartition(distances, -selected_topk_num)[-selected_topk_num:]

        centroid_class_data, centroid_class_label = (
            disagree_class_data[top_idx],
            disagree_class_label[top_idx],
        )
        centroid_data.extend(centroid_class_data)
        centroid_labels.extend(centroid_class_label)

    centroid_data = np.array(centroid_data)

    # (3) train Mp: Train Pp=Mp(X_mix), Loss=CrossEntropy(Pp, Y_mix)
    working_mix_data = np.concatenate([aux_data, selected_data, centroid_data], axis=0)
    working_mix_labels = np.concatenate([aux_labels, selected_labels, centroid_labels], axis=0)
    working_mix_labels_onehot = np.eye(num_classes)[working_mix_labels]

    working_mix_dataset = CustomDataset(working_mix_data, working_mix_labels_onehot)
    working_mix_dataloader_shuffled = DataLoader(
        working_mix_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True
    )

    model_train(
        working_mix_dataloader_shuffled,
        working_model,
        working_opt,
        working_criterion,
        alpha,
        args,
        device=device,
        save_path=working_model_save_path,
    )

    # 2. 通过 Mp 获取 D_disa_top={X_disa_top, P_disa_mix}, Train Pt=Mt(X_disa_top+Xa+Xs), Loss=CrossEntropy(Pt, P_disa_top)
    # (1) 获取Pp_disa: 通过 P_disa=Mp(D_disa)
    disagree_dataset = CustomDataset(disagree_data, teacher_disagree_predicts)
    disagree_dataloader = DataLoader(disagree_dataset, batch_size=args.batch_size, shuffle=False)

    working_disagree_predicts, working_disagree_probs = working_model_forward(disagree_dataloader, working_model)

    # (2) 获取Pp_disa_top: 获取max(P_disa), sample D_disa的top 10%数据
    working_disagree_probs_max = np.max(working_disagree_probs, axis=-1)  # [N]
    disagree_topk_num = len(working_disagree_predicts) // 10
    top_idx = np.argpartition(working_disagree_probs_max, -disagree_topk_num)[-disagree_topk_num:]

    disagree_data_top = disagree_data[top_idx]
    working_disagree_probs_top = working_disagree_probs[top_idx]

    # (3) 获取 P_disa_mix: P_disa_mix = (Pp_disa_top + Pt_disa_top) / 2
    teacher_disagree_probs_top = teacher_disagree_probs[top_idx]
    disagree_probs_top = (working_disagree_probs_top + teacher_disagree_probs_top) / 2

    # (4) P_disa_mix进行锐化
    temperature = 0.5
    disagree_probs_top_sharpen = np.pow(disagree_probs_top, 1. / temperature)
    disagree_probs_top_sharpen /= np.sum(disagree_probs_top_sharpen, axis=-1, keepdims=True)

    # (5) 获取 Dt_mix: Xt_mix = X_disa_top + Xa + Xs; Yt_mix = P_disa_mix + Ya + Ys
    teacher_mix_data = np.concatenate([disagree_data_top, aux_data, selected_data], axis=0)
    aux_labels_onehot = np.eye(num_classes)[aux_labels]
    selected_labels_onehot = np.eye(num_classes)[selected_labels]
    teacher_mix_labels = np.concatenate([disagree_probs_top_sharpen, aux_labels_onehot, selected_labels_onehot], axis=0)

    teacher_mix_dataset = CustomDataset(teacher_mix_data, teacher_mix_labels)
    teacher_mix_dataloader_shuffled = DataLoader(teacher_mix_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    # (6) train Mt: Pt=Mt(Da+Ds+D_disa_top)，Loss=CrossEntropy(Pt, Pconf)
    model_train(
        teacher_mix_dataloader_shuffled,
        teacher_model,
        teacher_opt,
        teacher_criterion,
        alpha,
        args,
        device=device,
        save_path=teacher_model_save_path
    )

    # 3. 获取Dconf 用与adapt: Dconf从Ds中随机抽取20%数据
    select_data_len = len(selected_labels)
    sample_size = select_data_len // 5
    random_sample_idx = np.random.choice(np.arange(select_data_len), size=sample_size, replace=False)
    conf_data = selected_data[random_sample_idx]
    conf_labels = selected_labels[random_sample_idx]
    conf_dataset = CustomDataset(conf_data, conf_labels)

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
    device,
    args,
):
    # 1. 构造Dts融合数据集 Dp_mix: (Dts, D_aug), 进行mix up
    # (1) 构造 Dp: Dp={Xts, Pp}, Pp = Mp(Xts)
    inc_data, inc_labels = inc_dataset.data, inc_dataset.label
    inc_predicts, inc_probs = working_model_forward(inc_dataloader, working_model)

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
        device=device,
        save_path=teacher_model_save_path
    )

    # 3. 重新构造 Dts融合数据集 Dp_mix
    # (1) 构造 Dp: Dp={Xts, Pp}, Pp = Mp(Xts)
    inc_predicts_new, inc_probs_new = working_model_forward(
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
        device=device,
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
        "working_model_path": os.path.join(ckpt_dir, "working_model_1", "p1_checkpoint.pth"),
        "working_model_repair_save_path": os.path.join(
            ckpt_dir, "working_model_repair"
        ),
        "working_model_adapt_save_path": os.path.join(ckpt_dir, "working_model_adapt"),
        "lip_teacher_model_path": os.path.join(ckpt_dir, "teacher_model_0", "t0_checkpoint.pth"),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 2. load model
    # (1) load working model
    if args.model == "resnet18":
        working_model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif args.model == "vgg16":
        working_model = vgg16_bn_lth(num_classes=num_classes)

    working_opt = optim.Adam(
        working_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    working_criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(working_model_path)
    working_model.load_state_dict(checkpoint, strict=False)
    # working_model.load_state_dict(checkpoint['state_dict'], strict=False)

    # (2) load lip_teacher model, t0的情况重新训练
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    lip_teacher_model = SimpleLipNet(resnet, 512, num_classes)
    teacher_opt = optim.Adam(
        lip_teacher_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    teacher_criterion = nn.CrossEntropyLoss()

    if os.path.exists(lip_teacher_model_path):
        checkpoint = torch.load(lip_teacher_model_path)
        lip_teacher_model.load_state_dict(checkpoint, strict=False)
    else:
        # t0 的情况下，使用D0数据重新训练 lip_teacher model
        print('Teacher model pth: %s not exist, only train T0, if not T0 then stop!' % lip_teacher_model_path)
        data_dir = dataset_paths[args.dataset]
        train_dataset, train_dataloader = get_dataset_loader(args.dataset, 'train', data_dir, args.batch_size,
                                                             num_classes=num_classes, drop_last=True, shuffle=True)

        lip_teacher_model_dir = os.path.dirname(lip_teacher_model_path)
        model_train(train_dataloader, lip_teacher_model, teacher_opt, teacher_criterion, alpha, args,
                    save_path=lip_teacher_model_dir)

    # todo temp code 待最后删除 训练 Mt0 Mt1 Mp0 Mp1
    # -------------------------------temp code start------------------------------------------------------------
    # # # (1) D0 数据训练Mt0
    # data_dir = dataset_paths[args.dataset]
    # test_dataset, test_dataloader = get_dataset_loader(
    #     "test", data_dir, args.batch_size, shuffle=False
    # )
    # inc_dataset, inc_dataloader = get_dataset_loader(
    #     "inc", data_dir, args.batch_size, shuffle=False
    # )
    # aux_dataset, aux_dataloader = get_dataset_loader(
    #     "aux", data_dir, args.batch_size, shuffle=False
    # )
    # curr_dir = os.path.dirname(os.path.abspath(__file__))
    # par_dir = os.path.dirname(curr_dir)
    # ckpt_dir = os.path.join(par_dir, "ckpt", args.dataset)
    #
    # #  D0
    # train_data = np.load(os.path.join(data_dir, 'cifar10_train_data.npy'))
    # train_labels = np.load(os.path.join(data_dir, 'cifar10_train_labels.npy'))
    # train_labels_onehot = np.eye(num_classes)[train_labels]
    #
    # train_dataset = CustomDataset(train_data, train_labels_onehot)
    # train_dataloader = DataLoader(train_dataset, args.batch_size, drop_last=True, shuffle=True)
    # # train teacher_model t0
    # # model_train(train_dataloader, lip_teacher_model, teacher_opt, teacher_criterion, alpha, args, save_path=ckpt_dir)
    # # train working_model p0
    # # model_train(train_dataloader, working_model, working_opt, working_criterion, alpha, args, save_path=ckpt_dir)
    #
    # # print("---------------------test t0------------------------------")
    # # model_test(test_dataset, test_dataloader, lip_teacher_model, device=device, teacher_model=True)
    # # print("---------------------test p0------------------------------")
    # # model_test(test_dataset, test_dataloader, working_model, device=device)
    #
    # # # (2) Da + Dtr 数据, 训练Mt0, 获得Mt1
    # aux_data, aux_labels = aux_dataset.data, aux_dataset.label
    # aux_labels_onehot = np.eye(num_classes)[aux_labels]
    # inc_data, inc_labels = inc_dataset.data, inc_dataset.label
    # inc_labels_onehot = np.eye(num_classes)[inc_labels]
    # aux_inc_data = np.concatenate([aux_data, inc_data], axis=0)
    # aux_inc_labels = np.concatenate([aux_labels_onehot, inc_labels_onehot], axis=0)
    # aux_inc_dataset = CustomDataset(aux_inc_data, aux_inc_labels)
    #
    # aux_inc_dataloader = DataLoader(aux_inc_dataset, args.batch_size, drop_last=True, shuffle=True)
    #
    # # train teacher_model t1
    # # model_train(aux_inc_dataloader, lip_teacher_model, teacher_opt, teacher_criterion, alpha, args, device=device,
    # #             save_path=ckpt_dir)
    #
    # # train working_model p1 Dtr
    # inc_dataset = CustomDataset(inc_data, inc_labels_onehot)
    # inc_dataloader = DataLoader(inc_dataset, args.batch_size, drop_last=True, shuffle=True)
    # # model_train(inc_dataloader, working_model, working_opt, working_criterion, alpha, args, device=device,
    # # save_path=ckpt_dir)
    #
    # # print("---------------------test t1------------------------------")
    # # model_test(test_dataset, test_dataloader, lip_teacher_model, device=device, teacher_model=True)
    # # print("---------------------test p1------------------------------")
    # # model_test(test_dataset, test_dataloader, working_model, device=device)

    # -------------------------------temp code end------------------------------------------------------------

    # 3. 迭代修复过程
    # (1) 构造修复过程数据集: Dtr、 Da、Dts
    data_dir = dataset_paths[args.dataset]
    inc_dataset, inc_dataloader = get_dataset_loader(
        args.dataset, "inc", data_dir, args.batch_size, shuffle=False
    )
    aux_dataset, aux_dataloader = get_dataset_loader(
        args.dataset, "aux", data_dir, args.batch_size, shuffle=False
    )
    test_dataset, test_dataloader = get_dataset_loader(
        args.dataset,"test", data_dir, args.batch_size, shuffle=False
    )

    # (2) 测试修复前 Dts 在 Mp 的表现
    print("---------------------working model test before------------------------------")
    working_model_test_before = model_test(test_dataset, test_dataloader, working_model, device=device)
    print("---------------------teacher model test before------------------------------")
    teacher_model_test_before = model_test(test_dataset, test_dataloader, lip_teacher_model, device=device,
                                           teacher_model=True)

    # (3) 迭代修复过程：根据 Dtr 迭代 Mp 、 Mt
    conf_dataset = None
    for i in range(repair_iter_num):
        print('-----------repair iterate %d ----------------------' % i)
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
            device,
            args,
        )

    # 4. 测试修复后 Dts 在 Mp 的表现
    working_model_after_repair = model_test(test_dataset, test_dataloader, working_model, device=device)
    teacher_model_after_repair = model_test(test_dataset, test_dataloader, lip_teacher_model, device=device,
                                            teacher_model=True)

    # 5. 迭代测试数据适应过程
    # (1) 构造适应过程数据：Dts, D_aug:  = Da + Dconf
    aux_data, aux_labels = aux_dataset.data, aux_dataset.label
    conf_data, conf_labels = conf_dataset.data, conf_dataset.label

    aug_data = np.concatenate([aux_data, conf_data], axis=0)
    aug_labels = np.concatenate([aux_labels, conf_labels], axis=0)
    aug_labels_onehot = np.eye(num_classes)[aug_labels]

    # (2) 迭代测试数据适应过程：根据 混合的Dts 迭代 Mp 和 Mt
    for i in range(adapt_iter_num):
        print('-----------adapt iterate %d ----------------------' % i)
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
            aug_labels_onehot,
            test_dataset,
            test_dataloader,
            device,
            args,
        )

    # 6. 测试适应后 Dts 在 Mp 的表现
    working_model_after_adapt = model_test(test_dataset, test_dataloader, working_model, device=device)
    teacher_model_after_adapt = model_test(test_dataset, test_dataloader, lip_teacher_model, device=device,
                                           teacher_model=True)

    print("---------------------working model test before------------------------------")
    print(working_model_test_before)
    print("---------------------teacher model test before------------------------------")
    print(teacher_model_test_before)

    print("---------------------working model test after repair------------------------------")
    print(working_model_after_repair)
    print("---------------------teacher model test after repair------------------------------")
    print(teacher_model_after_repair)

    print("---------------------working model test after adapt-------------------------------")
    print(working_model_after_adapt)
    print("---------------------teacher model test after adapt-------------------------------")
    print(teacher_model_after_adapt)

    logging.basicConfig(filename="../logs/core_execution.log", level=logging.INFO)
    logging.info(f"Test results before repair: {working_model_test_before}")
    logging.info(f"Test results after repair: {working_model_after_repair}")
    logging.info(f"Test results after adaptation: {working_model_after_adapt}")


if __name__ == "__main__":
    try:
        parse_args = parse_args()
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    execute(parse_args)

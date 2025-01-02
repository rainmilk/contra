import copy
import os
from functools import partial

from torch import nn
import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset_loader, MixupDataset, NormalizeDataset
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from configs import settings
from train_test import model_test, model_forward, model_train
import torchvision.transforms.v2 as transforms

import logging


def sharpen(probs, temperature=1.0, axis=-1):
    probs = probs ** (1.0 / temperature)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def label_smooth(labels, num_classes, gamma=0.0):
    return np.diag(np.ones(num_classes) - gamma)[labels] + gamma / num_classes


def auto_mixup(images, labels=None, alpha=0.75):
    nb_images = len(images)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    lbd = beta_dist.sample(sample_shape=(nb_images,))
    idx = lbd < 0.5
    lbd[idx] = 1 - lbd[idx]
    perm_idx = torch.randperm(nb_images)
    lbd_img = lbd.reshape((-1, 1, 1, 1))
    mixed_img = lbd_img * images + (1 - lbd_img) * images[perm_idx]
    if labels is not None:
        lbd_label = lbd.reshape((-1, 1))
        mixed_labels = lbd_label * labels + (1 - lbd_label) * labels[perm_idx]

    return mixed_img if labels is None else (mixed_img, mixed_labels)


def mixup_img(images1, images2, alpha=0.75):
    nb_images = len(images1)

    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    lbd = beta_dist.sample(sample_shape=(nb_images,))
    idx = lbd < 0.5
    lbd[idx] = 1 - lbd[idx]
    rnd_idx = torch.randint(low=0, high=len(images2), size=(nb_images,))
    lbd_img = lbd.reshape((-1, 1, 1, 1))
    mixed_img = lbd_img * images1 + (1 - lbd_img) * images2[rnd_idx]

    return mixed_img


def mixup(images1, labels1, images2, labels2, alpha=0.75):
    nb_images = len(images1)
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    lbd = beta_dist.sample(sample_shape=(nb_images,))
    idx = lbd < 0.5
    lbd[idx] = 1 - lbd[idx]
    rnd_idx = torch.randint(low=0, high=len(images2), size=(nb_images,))
    lbd_img = lbd.reshape((-1,1,1,1))
    mixed_img = lbd_img * images1 + (1 - lbd_img) * images2[rnd_idx]
    lbd_label = lbd.reshape((-1, 1))
    mixied_label = lbd_label * labels1 + (1 - lbd_label) * labels2[rnd_idx]

    return mixed_img, mixied_label


def mix_up_dataloader(
    first_data,
    first_probs,
    second_data,
    second_probs,
    batch_size,
    alpha=1.0,
    transforms=None,
    shuffle=True,
    first_max=True
):
    mixed_dataset = MixupDataset(
        data_pair=(first_data, second_data),
        label_pair=(first_probs, second_probs),
        mixup_alpha=alpha,
        transforms=transforms,
        first_max=first_max
    )
    return DataLoader(mixed_dataset, batch_size, drop_last=False, shuffle=shuffle)


def model_distill(model_teacher, model_student, epoch, data_loader,
                  transforms, criterion, optimizer, device, temperature=1):
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1} Distillation") as pbar:
        model_teacher.eval()
        model_student.train()
        for i, (image, target) in enumerate(data_loader):
            # predictions on data augmentation
            img_aug = transforms(image).to(device)
            # image_mixup = auto_mixup(image, None, alpha=0.75, device=device)
            pred_ul = model_teacher(img_aug)
            pred_ul = F.softmax(pred_ul/temperature, dim=1)

            optimizer.zero_grad()
            pred_infer = model_student(img_aug)
            loss = criterion(pred_infer, pred_ul)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)


def mria_train(args):
    num_classes = settings.num_classes_dict[args.dataset]
    # kwargs = parse_kwargs(args.kwargs)
    case = settings.get_case(args.forget_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = os.path.join(settings.root_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, "core_execution.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    learning_rate = args.learning_rate
    lr_student = args.lr_student
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    num_epochs = args.num_epochs
    uni_name = args.uni_name

    model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="ul", unique_name=uni_name,
    )

    model_save_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="restore", unique_name=uni_name,
    )

    model_student_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="student", unique_name=uni_name,
    )

    train_model_path = settings.get_ckpt_path(
        args.dataset, case, args.model, model_suffix="train", unique_name=uni_name,
    )

    backbone = load_custom_model(args.model, num_classes, load_pretrained=False)
    model_student = ClassifierWrapper(backbone, num_classes)
    checkpoint = torch.load(model_path)
    model_student.load_state_dict(checkpoint, strict=False)
    optimizer, lr_scheduler = create_optimizer_scheduler(
        optimizer_type,
        model_student.parameters(),
        num_epochs,
        learning_rate,
        weight_decay,
    )

    model_teacher = copy.deepcopy(model_student)
    model_teacher.to(device=device)
    model_student.to(device=device)
    # switch to train mode

    loss_fn = nn.CrossEntropyLoss()

    train_data, train_labels, train_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, forget_loader = get_dataset_loader(
        args.dataset,
        ["forget"],
        [case],
        batch_size=args.batch_size,
        shuffle=False,
    )

    _, _, data_loader = get_dataset_loader(
        args.dataset,
        ["test", "forget"],
        [None, case],
        batch_size=args.batch_size,
        shuffle=True,
    )

    model_test(forget_loader, model_student, device)
    auto_mix = partial(auto_mixup, labels=None, alpha=0.75)
    temperature = args.temperature

    for i in tqdm(range(num_epochs), desc="MRIA"):
        # Distillation Stage
        print(f"MRIA Epoch {i}: Distillation Stage")
        for epoch in range(args.distill_epochs):
            model_distill(model_student, model_student, epoch, data_loader,
                          auto_mix, loss_fn, optimizer, device, temperature)
            model_test(forget_loader, model_student, device)
            lr_scheduler.step(epoch)


        # Alignment Stage
        top_conf = args.top_conf
        iters = 5

        print(f"MRIA Epoch {i}: Alignment Stage")
        def get_conf_data_loader(train_probs, infer_probs):
            joint_probs = torch.as_tensor(train_probs * infer_probs)
            nb_samples = len(train_probs)
            k = round(top_conf * nb_samples / num_classes)
            _, conf_topk = torch.topk(joint_probs, k=k, dim=0)
            conf_topk = conf_topk.numpy().flatten()
            conf_labels = np.tile(np.arange(num_classes), k)
            conf_data = train_data[conf_topk]
            # agree_labels = np.argmax(conf_probs, axis=-1)
            conf_agree_probs = label_smooth(conf_labels, num_classes, gamma=args.ls_gamma)
            conf_dataset = NormalizeDataset(conf_data, conf_agree_probs)
            return DataLoader(conf_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)


        model_teacher = model_student
        train_predicts, train_probs = model_forward(train_loader, model_teacher)
        for _ in range(args.align_epochs):
            infer_predicts, infer_probs = model_forward(train_loader, model_student)
            conf_data_loader = get_conf_data_loader(train_probs, infer_probs)

            model_train(
                conf_data_loader,
                model_teacher,
                ul_optimizer,
                ul_lr_scheduler,
                loss_fn,
                iters,
                args,
                device=device,
            )
            model_test(forget_loader, model_teacher, device)

            train_predicts, train_probs = model_forward(train_loader, model_teacher)
            conf_data_loader = get_conf_data_loader(train_probs, infer_probs)
            model_train(
                conf_data_loader,
                model_student,
                optimizer,
                lr_scheduler,
                loss_fn,
                iters,
                args,
                device=device,
            )
            model_test(forget_loader, model_student, device)

    print("Teacher Model Performance:")
    model_test(train_loader, model_student, device)
    state = model_student.state_dict()
    torch.save(state, model_save_path)

    print("Student Model Performance:")
    model_test(train_loader, model_student, device)
    state = model_student.state_dict()
    torch.save(state, model_student_path)

    return model_student, model_student


if __name__ == "__main__":
    args = arg_parser.parse_args()
    mria_train(args)

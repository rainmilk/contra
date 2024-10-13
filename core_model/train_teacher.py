import os
import shutil

from core import train_teacher_model
from torch import nn
from configs import settings

from args_paser import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper
from core_model.dataset import get_dataset_loader
import torch
from train_test_utils import train_model

if __name__ == "__main__":
    args = parse_args()

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    num_epochs =args.num_epochs
    batch_size = args.batch_size
    model_suffix = "teacher_restore" if args.model_suffix is None else args.model_suffix
    step = getattr(args, "step", 0)

    model_name = args.model
    noise_ratio = args.noise_ratio
    noise_type = args.noise_type
    balanced = args.balanced
    dataset = args.dataset
    spec_norm = not args.no_spnorm
    uni_name = args.uni_name

    pretrain_case = "pretrain"
    model_suffix = "teacher_restore"
    model_p0_path = settings.get_ckpt_path(
        dataset,
        pretrain_case,
        model_name,
        model_suffix,
        step=0
    )

    if uni_name is None:
        num_classes = settings.num_classes_dict[dataset]
        backbone = load_custom_model(model_name, num_classes, load_pretrained=True)
        # features = backbone.fc.in_features
        # backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        # lip_teacher_model = SimpleLipNet(backbone, features, num_classes, spectral_norm=spec_norm)
        lip_teacher_model = ClassifierWrapper(backbone, num_classes, spectral_norm=spec_norm)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lip_teacher_model.to(device)

        # 根据用户选择的优化器初始化
        teacher_opt, teacher_lr_scheduler = create_optimizer_scheduler(
            optimizer_type=optimizer_type,
            parameters=lip_teacher_model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=num_epochs,
        )

        teacher_criterion = nn.CrossEntropyLoss()

        train_teacher_model(args, step, num_classes, lip_teacher_model, teacher_opt, teacher_lr_scheduler,
                            teacher_criterion, model_p0_path, test_per_it=1)

        # case = settings.get_case(args.noise_ratio, args.noise_type, args.balanced)
        # train_data, train_labels, train_dataloader = get_dataset_loader(
        #     args.dataset,
        #     "train",
        #     case,
        #     step,
        #     None,
        #     None,
        #     args.batch_size,
        #     num_classes=num_classes,
        #     drop_last=False,
        #     shuffle=True,
        #     onehot_enc=False,
        # )
        #
        # test_data, test_labels,  test_dataloader = get_dataset_loader(
        #     args.dataset,
        #     "test",
        #     case,
        #     None,
        #     mean=None,
        #     std=None,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        # )
        #
        # lip_teacher_model = train_model(
        #     lip_teacher_model,
        #     num_classes,
        #     teacher_opt,
        #     teacher_lr_scheduler,
        #     train_dataloader,
        #     test_dataloader,
        #     epochs=args.num_epochs
        # )
    else:
        case = settings.get_case(noise_ratio, noise_type, balanced)
        copy_model_p0_path = settings.get_ckpt_path(
            dataset,
            case,
            model_name,
            model_suffix,
            step=step,
            unique_name=uni_name,
        )
        if os.path.exists(model_p0_path):
            subdir = os.path.dirname(copy_model_p0_path)
            os.makedirs(subdir, exist_ok=True)
            shutil.copy(model_p0_path, copy_model_p0_path)
            print(f"Copy {model_p0_path} to {copy_model_p0_path}")
        else:
            raise FileNotFoundError(model_p0_path)

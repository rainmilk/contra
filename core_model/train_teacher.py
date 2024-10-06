import os
from core import train_teacher_model
from torch import nn
from configs import settings

from args_paser import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper

if __name__ == '__main__':
    args = parse_args()

    learning_rate = getattr(args, "learning_rate", 0.001)
    weight_decay = getattr(args, "weight_decay", 5e-4)
    optimizer_type = getattr(args, "optimizer", "adam")
    num_epochs = getattr(args, "num_epochs", 50)
    model_name = getattr(args, "model", 'resnet18')
    batch_size = getattr(args, "batch_size", 64)
    model_suffix = getattr(args, "model_suffix", "teacher_restore")
    step = getattr(args, "step", 0)

    noise_ratio = args.noise_ratio
    noise_type = args.noise_type
    balanced = args.balanced
    dataset = args.dataset
    uni_name = args.uni_name

    num_classes = settings.num_classes_dict[dataset]
    backbone = load_custom_model(model_name, num_classes, load_pretrained=True)
    features = backbone.fc.in_features
    backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    lip_teacher_model = SimpleLipNet(backbone, features, num_classes)
    # lip_teacher_model = ClassifierWrapper(backbone, num_classes)

    # 根据用户选择的优化器初始化
    teacher_opt, teacher_lr_scheduler = create_optimizer_scheduler(
        optimizer_type, lip_teacher_model.parameters(), num_epochs, learning_rate, weight_decay
    )
    teacher_criterion = nn.CrossEntropyLoss()
    case = settings.get_case(noise_ratio, noise_type, balanced)
    model_paths = settings.get_ckpt_path(dataset, case, model_name, model_suffix=model_suffix,
                                         step=step, unique_name=uni_name)

    train_teacher_model(args, step, num_classes, lip_teacher_model, teacher_opt,
                        teacher_lr_scheduler, teacher_criterion, model_paths,
                        test_per_it=1)

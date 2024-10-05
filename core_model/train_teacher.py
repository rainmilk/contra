import os
from core import train_teacher_model, dataset_paths, get_model_paths
from torch import nn
from dataset import get_dataset_loader

from main import parse_args, parse_kwargs
from lip_teacher import SimpleLipNet
from optimizer import create_optimizer_scheduler
from custom_model import load_custom_model, ClassifierWrapper

if __name__ == '__main__':
    args = parse_args()

    learning_rate = getattr(args, "learning_rate", 0.001)
    weight_decay = getattr(args, "weight_decay", 5e-4)
    optimizer_type = getattr(args, "optimizer", "adam")
    num_epochs = getattr(args, "num_epochs", 50)
    num_classes = getattr(args, "num_classes", 37)
    model_name = getattr(args, "model", 'resnet18')
    batch_size = getattr(args, "batch_size", 256)
    dataset = getattr(args, "dataset", "cifar-10")

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
    data_dir = dataset_paths[dataset]
    model_paths = get_model_paths(model_name, dataset)
    lip_teacher_model_dir = os.path.dirname(model_paths["lip_teacher_model_path"])

    train_teacher_model(args, data_dir, num_classes, lip_teacher_model, teacher_opt,
                        teacher_lr_scheduler, teacher_criterion, lip_teacher_model_dir,
                        test_per_it=1)

import torch
import torchvision.models as models
from torch import nn, optim
from nets import *
from nets.cifar_10.vgg import VGG
from nets.cifar_10.resnet import ResNet


class ModelUtils:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def get_criterion_and_optimizer(self, model, lr=0.001, **kwargs):
        """
        获取损失函数和优化器。
        默认为 CrossEntropyLoss 和 Adam 优化器。
        可以通过 kwargs 传递更多的参数，例如 weight_decay, momentum 等。
        """
        criterion = nn.CrossEntropyLoss()

        # 从 kwargs 中获取可选参数
        weight_decay = kwargs.get("weight_decay", 0)  # 默认 weight_decay 为 0
        momentum = kwargs.get("momentum", 0)  # 针对 SGD 优化器的参数
        optimizer_type = kwargs.get("optimizer", "adam").lower()  # 默认为 Adam 优化器

        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        return criterion, optimizer

    def create_model(self):
        """
        根据指定的模型名称创建模型，支持 ResNet18 和 VGG16。
        """
        if self.model_name == "resnet18":
            # 修改最后一层，适应 num_classes
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif self.model_name == "vgg16":
            # 加载预训练模型，但不传递 num_classes 参数
            model = models.vgg16(pretrained=True)
            # 修改 VGG16 的最后一层，使其适应 num_classes
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, self.num_classes
            )

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        print(
            f"Created model: {self.model_name} with {self.num_classes} output classes."
        )
        return model

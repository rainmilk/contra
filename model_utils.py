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

    def get_criterion_and_optimizer(self, model, lr=0.001):
        """
        获取损失函数和优化器。
        默认为 CrossEntropyLoss 和 Adam 优化器。
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return criterion, optimizer

    def create_model(self):
        """
        根据指定的模型名称创建模型，支持 ResNet18 和 VGG16。
        """
        if self.model_name == "resnet18":
            
            # 修改最后一层，适应num_classes
            model = models.resnet18(pretrained=False, num_classes=self.num_classes)

        elif self.model_name == "vgg16":
            model = VGG('VGG16')
    
        # elif self.model_name == "vgg16":
        #     model = models.vgg16(pretrained=True)  # 尝试使用预训练权重
        #     # 修改 VGG16 的最后一层，使其适应 num_classes
        #     model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        print(f"Created model: {self.model_name} with {self.num_classes} output classes.")
        return model

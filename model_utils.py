import torch
import torchvision
from torch import nn, optim


class ModelUtils:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def create_resnet_model(self):
        if self.model_name == "resnet18":
            model = torchvision.models.resnet18(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.model_name == "vgg16":
            model = torchvision.models.vgg16(
                pretrained=False, num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_criterion_and_optimizer(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return criterion, optimizer

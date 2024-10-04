import torch.nn as nn
import torch
from torchvision import models
from core_model.wideresidual import wideresnet
from core_model.resnet import resnet18


def load_custom_model(model_name, num_classes, load_pretrained=False, ckpt_path=None):
    weights = None
    if model_name == "resnet18":
        if load_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights, num_classes=num_classes)
    elif model_name == "vgg19":
        if load_pretrained:
            weights = models.VGG19_BN_Weights.DEFAULT
        model = models.vgg19_bn(weights=weights, num_classes=num_classes)
    elif model_name == "cifar-resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_name == "cifar-wideresnet40":
        model = wideresnet(num_classes=num_classes)
    else:
        model = None

    if model and ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint, strict=False)

    return model


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone, num_classes, freeze_weights=False,
                 batchnorm_blocks=-1, spectral_norm=False):
        super(ClassifierWrapper, self).__init__()

        # Freezing the weights
        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove the final layer
        all_modules = list(backbone.children())
        features = all_modules[-1].in_features

        modules = [*all_modules[:-1], nn.Flatten()]
        if batchnorm_blocks >= 0:
            modules += [
                *[nn.ReLU(), nn.BatchNorm1d(features), nn.Linear(features, features)] * batchnorm_blocks,
                nn.ReLU(), nn.BatchNorm1d(features)]
        self.feature_model = nn.Sequential(*modules)
        if spectral_norm:
            self.apply(self.add_spectral_norm_)

        self.fc = nn.Linear(features, num_classes)

    def forward(self, x, output_emb=False):
        emb = self.feature_model(x)
        outputs = self.fc(emb)
        if output_emb:
            return outputs, emb

        return outputs

    def add_spectral_norm_(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.utils.spectral_norm(m)

import torch.nn as nn
import torch
from torchvision import models


def load_custom_model(model_name, num_classes, load_pretrained=True, ckpt_path=None):
    weights = None
    if model_name == "resnet18":
        if load_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet50":
        if load_pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(num_classes=num_classes)
    elif model_name == "resnet101":
        if load_pretrained:
            weights = models.ResNet101_Weights.DEFAULT
            model = models.resnet101(weights=weights)
        else:
            model = models.resnet101(num_classes=num_classes)
    elif model_name == "wideresnet50":
        if load_pretrained:
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            model = models.wide_resnet50_2(weights=weights)
        else:
            model = models.wide_resnet50_2(num_classes=num_classes)
    elif model_name == "efficientnet_s":
        if load_pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            model = models.efficientnet_v2_s(weights=weights)
        else:
            model = models.efficientnet_v2_s(num_classes=num_classes)
    elif model_name == "efficientnet_m":
        if load_pretrained:
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
            model = models.efficientnet_v2_m(weights=weights)
        else:
            model = models.efficientnet_v2_m(num_classes=num_classes)
    elif model_name == "efficientnet_l":
        if load_pretrained:
            weights = models.EfficientNet_V2_L_Weights.DEFAULT
            model = models.efficientnet_v2_l(weights=weights)
        else:
            model = models.efficientnet_v2_l(num_classes=num_classes)
    elif model_name == "efficientnet_b3":
        if load_pretrained:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            model = models.efficientnet_b3(weights=weights)
        else:
            model = models.efficientnet_b3(num_classes=num_classes)
    elif model_name == "efficientnet_b7":
        if load_pretrained:
            weights = models.EfficientNet_B7_Weights.DEFAULT
            model = models.efficientnet_b7(weights=weights)
        else:
            model = models.efficientnet_b7(num_classes=num_classes)
    elif model_name == "vit_b_16":
        if load_pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
            model = models.vit_b_16(weights=weights)
        else:
            model = models.vit_b_16(num_classes=num_classes)
    elif model_name == "vit_b_32":
        if load_pretrained:
            weights = models.ViT_B_32_Weights.DEFAULT
            model = models.vit_b_32(weights=weights)
        else:
            model = models.vit_b_32(num_classes=num_classes)
    elif model_name == "swin_t":
        if load_pretrained:
            weights = models.Swin_V2_T_Weights.DEFAULT
            model = models.swin_v2_t(weights=weights)
        else:
            model = models.swin_v2_t(num_classes=num_classes)
    elif model_name == "swin_s":
        if load_pretrained:
            weights = models.Swin_V2_S_Weights.DEFAULT
            model = models.swin_v2_s(weights=weights)
        else:
            model = models.swin_v2_s(num_classes=num_classes)
    elif model_name == "maxvit_t":
        if load_pretrained:
            weights = models.MaxVit_T_Weights.DEFAULT
            model = models.maxvit_t(weights=weights)
        else:
            model = models.maxvit_t(num_classes=num_classes)
    elif model_name == "vgg16":
        if load_pretrained:
            weights = models.VGG16_Weights.DEFAULT
            model = models.vgg16(weights=weights)
        else:
            model = models.vgg16(num_classes=num_classes)
    else:
        raise Exception(f"{model_name} is not supported")


    if model and ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint, strict=False)
        print('load worker model from :', ckpt_path)
    return model


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone, num_classes,
                 freeze_weights=False,
                 spectral_norm=False):
        super(ClassifierWrapper, self).__init__()

        # Freezing the weights
        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        all_modules = list(backbone.children())
        for i, module in enumerate(all_modules):
            if isinstance(module, nn.ModuleList):
                all_modules[i] = nn.Sequential(*module)

        self.feature_model = nn.Sequential(*all_modules[:-1], nn.Flatten())

        children = list(all_modules[-1].children())
        if len(children) > 1:
            feature_size = children[-1].in_features
            self.fc = nn.Sequential(*children[:-1], nn.Linear(feature_size, num_classes))
        else:
            feature_size = all_modules[-1].in_features
            self.fc = nn.Linear(feature_size, num_classes)

        if spectral_norm:
            self.apply(self._add_spectral_norm)

    def forward(self, x, output_emb=False):
        emb = self.feature_model(x)
        outputs = self.fc(emb)
        if output_emb:
            return outputs, emb

        return outputs

    def _add_spectral_norm(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.utils.spectral_norm(m)

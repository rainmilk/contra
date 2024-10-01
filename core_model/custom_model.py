import torch.nn as nn


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone, num_classes, freeze_weights=False):
        super(ClassifierWrapper, self).__init__()

        # Freezing the weights
        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove the final layer
        all_modules = list(backbone.children())
        self.feature_model = nn.Sequential(*all_modules[:-1])
        self.fc = nn.Linear(all_modules[-1].out_features, num_classes)

    def forward(self, x, output_emb=False):
        emb = self.feature_model(x)
        outputs = self.fc(emb)
        if output_emb:
            return outputs, emb

        return outputs

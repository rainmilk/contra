import torch.nn as nn


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone, num_classes, freeze_weights=False, batchnorm_blocks=-1):
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
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x, output_emb=False):
        emb = self.feature_model(x)
        outputs = self.fc(emb)
        if output_emb:
            return outputs, emb

        return outputs

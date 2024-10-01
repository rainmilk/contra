import torch.nn as nn


class CustomClassifier(nn.Module):
    def __init__(self, backbone, out_feature=512, freeze_weights=False, dropout=0.25):
        super(CustomClassifier, self).__init__()

        # Freezing the weights
        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove the final layer

        self.base_model = nn.Sequential(*list(backbone.children())[:-1])
        num_classes = backbone.children()[-1].out_features

        self.output_model = nn.Sequential(
            self.base_model,
            nn.Linear(self.base_model.children()[-1].out_features, out_feature)
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feature, num_classes)
        )

    def forward(self, x):
        outputs = self.output_model(x)
        outputs = self.classifier(outputs)
        return outputs
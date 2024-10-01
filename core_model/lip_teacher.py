import torch.nn as nn
import numpy as np
import torch
from torchvision import models


class SimpleLipNet(nn.Module):
    """
    Implementation of Lipschitz regularized network
    """

    def __init__(self, backbone, input_sz, output_sz):
        super(SimpleLipNet, self).__init__()

        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.apply(self.add_spectral_norm_)
        self.fc = nn.Linear(input_sz, output_sz)
        self.apply(self._spectral_init)

    # 使用谱归一化 (spectral_norm) 来规范化线性层和卷积层的权重
    def add_spectral_norm_(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.utils.spectral_norm(m)

    # 为线性层和卷积层进行 Xavier 初始化，使用 SVD 对权重进行缩放处理。
    def _spectral_init(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

            u, s, v = torch.svd(m.weight)
            m.weight.data = 1.0 * m.weight.data / s[0]

        elif isinstance(m, (nn.Conv2d)):
            torch.nn.init.xavier_normal_(m.weight)
            weight = torch.reshape(m.weight.data, (m.weight.data.shape[0], -1))
            u, s, v = torch.svd(weight)
            m.weight.data = m.weight.data / s[0]

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):  # [N,C,H,W]
        x = self.backbone(inputs)
        embedding = self.flatten(x)
        out = self.fc(embedding)
        if self.training:
            return out
        return embedding, out


if __name__ == "__main__":
    resnet = models.resnet18(pretrained=False, num_classes=512)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    print(resnet)
    simple_lip_net = SimpleLipNet(resnet, 512, 10)
    data = torch.tensor(np.random.randn(8, 3, 32, 32).astype(np.float32))
    label = torch.tensor(np.arange(0, 8))
    print(simple_lip_net)

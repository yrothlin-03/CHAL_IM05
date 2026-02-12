import torch
import torch.nn as nn

from .head import ClassificationHead
from .v3 import BackboneV3


class Model(nn.Module):
    def __init__(self, num_classes: int = 13, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = BackboneV3(pretrained=pretrained_backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.backbone.out_channels
        self.head = ClassificationHead(in_features=in_features, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        print(f"Backbone output shape: {x.shape}")
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
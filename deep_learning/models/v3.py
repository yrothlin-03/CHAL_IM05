import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights



class BackboneV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B7_Weights.DEFAULT if pretrained else None
        model = efficientnet_b7(weights=weights)
        self.features = model.features
        self.out_channels = model.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)
        return x
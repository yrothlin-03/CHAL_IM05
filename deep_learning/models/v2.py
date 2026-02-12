import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50


class BackboneV2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = backbone.fc.in_features 
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # self.head = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        # x = torch.flatten(x, 1)
        # x = self.head(x)
        return x
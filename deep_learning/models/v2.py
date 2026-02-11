import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50

# class Model(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         m = resnet50(weights=ResNet50_Weights.DEFAULT)
#         m.fc = nn.Linear(m.fc.in_features, num_classes)
#         self.model = m

#     def forward(self, x):
#         return self.model(x)


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
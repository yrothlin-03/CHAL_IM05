import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)
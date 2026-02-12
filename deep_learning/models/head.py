import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, dim_hidden=256, num_layer=2, num_classes=13):
        super().__init__()
        layers = [] 
        for i in range(num_layer): 
            layers.append(nn.Linear(in_features, dim_hidden))
            layers.append(nn.ReLU())
            in_features = dim_hidden
            
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)
                                        

    
    def forward(self, x):
        return self.layers(x)


class Head(nn.Module):
    def __init__(self, num_classes: int = 13):
        super().__init__()
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)   
        return self.classifier(x)


class ConvHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 13,
        num_layer: int = 2,
        dim_hidden: int = 256,
    ):
        super().__init__()

        layers = []
        in_ch = in_channels

        for _ in range(num_layer):
            layers.append(nn.Conv2d(in_ch, dim_hidden, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(dim_hidden))
            layers.append(nn.ReLU(inplace=True))
            in_ch = dim_hidden

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        
        x = self.classifier(x)       
        x = self.pool(x)              
        x = torch.flatten(x, 1)       
        return x
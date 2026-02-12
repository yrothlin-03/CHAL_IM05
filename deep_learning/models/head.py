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


        
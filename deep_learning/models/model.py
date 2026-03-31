import torch
import torch.nn as nn

from .backbone import Backbone
from .head import (
    HEAD,
    HEAD2,
    CNN_HEAD,
    TRANSFORMER_HEAD,
    QUERY_ATTENTION_HEAD
)


class Model(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 13, freeze_backbone: bool = False, pretrained: bool = False):
        super().__init__()

        self.backbone = Backbone(name=backbone_name, pretrained=pretrained)
        self.features_shape = self.backbone._get_features_shape()

        if freeze_backbone:
            self.freeze_backbone()

        # self.head = HEAD2(
        #     n_classes=num_classes,
        #     in_dim=self.features_shape,
        #     # n_layers=1,
        #     p=0.3
        # )

        self.head = HEAD(
            n_classes = num_classes,
            n_layers = 1,
            in_dim = self.features_shape,
            p = 0.3
        )

        # self.head = QUERY_ATTENTION_HEAD(
        #     n_classes=num_classes,
        #     in_dim=self.features_shape,
        #     d_model=256,
        #     n_queries=20,
        #     n_heads=4,
        #     p=0.3,
        # )

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("----" * 20)
        print("MODEL INFORMATIONS")
        print(f"Backbone         : {backbone_name}")
        print(f"Feature dim      : {self.features_shape}")
        print(f"Backbone params  : {backbone_params:,}")
        print(f"Head params      : {head_params:,}")
        print(f"Total params     : {total_params:,}")
        print(f"Trainable params : {trainable_params:,}")
        print(f"Frozen backbone  : {freeze_backbone}")
        print("----" * 20)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()  

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()


    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
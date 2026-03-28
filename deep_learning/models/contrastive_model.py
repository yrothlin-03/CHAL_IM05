import torch
import torch.nn as nn

from .backbone import Backbone


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, p: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        projection_dim: int = 128,
        projection_hidden_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = Backbone(name=backbone_name)
        self.features_shape = self.backbone._get_features_shape()

        if freeze_backbone:
            self.freeze_backbone()

        self.projector = ProjectionHead(
            in_dim=self.features_shape,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
            p=dropout,
        )

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("----" * 20)
        print("CONTRASTIVE MODEL INFORMATIONS")
        print(f"Backbone         : {backbone_name}")
        print(f"Feature dim      : {self.features_shape}")
        print(f"Projection dim   : {projection_dim}")
        print(f"Backbone params  : {backbone_params:,}")
        print(f"Projector params : {projector_params:,}")
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

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.backbone(x)
        z = self.projector(features)
        return z
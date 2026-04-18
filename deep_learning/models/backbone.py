import torch
import torch.nn as nn
from typing import Literal

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet169, DenseNet169_Weights,
    swin_v2_t, Swin_V2_T_Weights,
)

from transformers import ViTModel

from .cnn import CNN_Backbone
from .transformer import (
    transformer_backbone_tiny,
    transformer_backbone_small,
    transformer_backbone_base,
)


BackboneName = Literal[
    'RESNET_50',
    'RESNET_18',
    'RESNET_101',
    'EFFICIENTNET_B7',
    'EFFICIENTNET_V2_S',
    'EFFICIENTNET_V2_M',
    'VIT_B_16',
    'GOOGLE_VIT_B_16',
    'CONVNEXT_BASE',
    'CONVNEXT_TINY',
    'DENSENET_169',
    'SWIN_V2_T',
]


class Backbone(nn.Module):
    def __init__(self, name: BackboneName, pretrained: bool = False):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.backbone = self._load_backbone(name)
        self.features_shape = self._get_features_shape()

    def _load_backbone(self, name: str) -> nn.Module:

        if name == 'RESNET_18':
            weights = ResNet18_Weights.DEFAULT if self.pretrained else None
            model = resnet18(weights=weights)
            return nn.Sequential(*list(model.children())[:-1])

        elif name == 'RESNET_50':
            weights = ResNet50_Weights.DEFAULT if self.pretrained else None
            model = resnet50(weights=weights)
            return nn.Sequential(*list(model.children())[:-1])

        elif name == 'RESNET_101':
            weights = ResNet101_Weights.DEFAULT if self.pretrained else None
            model = resnet101(weights=weights)
            return nn.Sequential(*list(model.children())[:-1])

        elif name == 'EFFICIENTNET_B7':
            weights = EfficientNet_B7_Weights.DEFAULT if self.pretrained else None
            model = efficientnet_b7(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'EFFICIENTNET_V2_S':
            weights = EfficientNet_V2_S_Weights.DEFAULT if self.pretrained else None
            model = efficientnet_v2_s(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'EFFICIENTNET_V2_M':
            weights = EfficientNet_V2_M_Weights.DEFAULT if self.pretrained else None
            model = efficientnet_v2_m(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'VIT_B_16':
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if self.pretrained else None
            model = vit_b_16(weights=weights)
            model.heads = nn.Identity()
            return model

        elif name == 'GOOGLE_VIT_B_16':
            if self.pretrained:
                model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            else:
                model = ViTModel.from_pretrained("google/vit-base-patch16-224", ignore_mismatched_sizes=True)
            return model

        elif name == 'CONVNEXT_BASE':
            weights = ConvNeXt_Base_Weights.DEFAULT if self.pretrained else None
            model = convnext_base(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'CONVNEXT_SMALL':
            weights = ConvNeXt_Small_Weights.DEFAULT if self.pretrained else None
            model = convnext_small(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'CONVNEXT_TINY':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if self.pretrained else None
            model = convnext_tiny(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'DENSENET_169':
            weights = DenseNet169_Weights.DEFAULT if self.pretrained else None
            model = densenet169(weights=weights)
            return nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        elif name == 'SWIN_V2_T':
            weights = Swin_V2_T_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = swin_v2_t(weights=weights)
            model.head = nn.Identity()
            return model

        elif name == 'MY_CNN':
            return CNN_Backbone()
        
        elif name == 'MY_TRANSFORMER_TINY':
            return transformer_backbone_tiny()

        elif name == 'MY_TRANSFORMER_SMALL':
            return transformer_backbone_small()

        elif name == 'MY_TRANSFORMER_BASE':
            return transformer_backbone_base()  
        
        else:
            raise ValueError(f"Backbone inconnu: {name}")

    def _get_features_shape(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            y = self.backbone(x)
            if hasattr(y, "last_hidden_state"):
                y = y.last_hidden_state[:, 0, :]   # token CLS
            elif isinstance(y, (tuple, list)):
                y = y[0]
            if y.dim() > 2:
                y = torch.flatten(y, 1)
        return y.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if hasattr(x, "last_hidden_state"):
            x = x.last_hidden_state[:, 0, :]   # token CLS
        elif isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return x
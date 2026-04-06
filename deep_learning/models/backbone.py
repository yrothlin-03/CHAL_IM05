import torch
import torch.nn as nn
from typing import Literal

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    regnet_y_16gf, RegNet_Y_16GF_Weights,
    vit_b_16, ViT_B_16_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet169, DenseNet169_Weights,
    swin_t, Swin_T_Weights,
    swin_v2_t, Swin_V2_T_Weights,
)

from transformers import AutoModel


BackboneName = Literal[
    'RESNET_50',
    'RESNET_18',
    'RESNET_101',
    'EFFICIENTNET_B4',
    'EFFICIENTNET_B7',
    'EFFICIENTNET_V2_S',
    'EFFICIENTNET_V2_M',
    'EFFICIENTNET_V2_L',
    'REGNET',
    'VIT_B_16',
    'CONVNEXT_BASE',
    'CONVNEXT_SMALL',
    'CONVNEXT_TINY',
    'DENSENET_169',
    'SWIN_T',
    'SWIN_V2_T',
    'DINOV2_VITS14',
    'DINOV3_VITS16',
]


class DinoV3Wrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]

        raise RuntimeError("DINOv3 model output does not contain pooler_output or last_hidden_state.")


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

        elif name == 'EFFICIENTNET_B4':
            weights = EfficientNet_B4_Weights.DEFAULT if self.pretrained else None
            model = efficientnet_b4(weights=weights)
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

        elif name == 'EFFICIENTNET_V2_L':
            weights = EfficientNet_V2_L_Weights.DEFAULT if self.pretrained else None
            model = efficientnet_v2_l(weights=weights)
            return nn.Sequential(
                model.features,
                model.avgpool,
            )

        elif name == 'REGNET':
            weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1 if self.pretrained else None
            model = regnet_y_16gf(weights=weights)
            return nn.Sequential(
                model.trunk_output,
                model.avgpool,
            )

        elif name == 'VIT_B_16':
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if self.pretrained else None
            model = vit_b_16(weights=weights)
            model.heads = nn.Identity()
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

        elif name == 'SWIN_T':
            weights = Swin_T_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = swin_t(weights=weights)
            model.head = nn.Identity()
            return model

        elif name == 'SWIN_V2_T':
            weights = Swin_V2_T_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = swin_v2_t(weights=weights)
            model.head = nn.Identity()
            return model

        elif name == 'DINOV2_VITS14':
            if not self.pretrained:
                raise ValueError("DINOV2_VITS14 should be used with pretrained=True.")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            return model

        elif name == 'DINOV3_VITS16':
            if not self.pretrained:
                raise ValueError("DINOV3_VITS16 should be used with pretrained=True.")

            model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            return DinoV3Wrapper(model_name)

        else:
            raise ValueError(f"Backbone inconnu: {name}")

    def _get_features_shape(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            y = self.backbone(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            if y.dim() > 2:
                y = torch.flatten(y, 1)
        return y.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing DINOV3_VITS16 instantiation...")
    backbone = Backbone(name='DINOV3_VITS16', pretrained=True).to(device)
    backbone.eval()

    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        y = backbone(x)

    print(f"Backbone name   : {backbone.name}")
    print(f"Feature dim     : {backbone.features_shape}")
    print(f"Output shape    : {tuple(y.shape)}")
    print("DINOV3_VITS16 instantiation OK.")
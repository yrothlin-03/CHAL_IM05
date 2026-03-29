import torch
import torch.nn as nn


class HEAD(nn.Module):
    def __init__(self, n_classes: int, in_dim: int, n_layers: int, p: float = 0.3):
        super().__init__()

        layers = []
        dim = in_dim

        for i in range(n_layers - 1):
            next_dim = max(dim // 2, n_classes)  
            layers.append(nn.Linear(dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p))
            dim = next_dim

        layers.append(nn.Linear(dim, n_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
    



class HEAD2(nn.Module):
    def __init__(self, n_classes: int, in_dim: int, hidden_dim: int = 512, p: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.head(x)
    

    

class HEAD3(nn.Module):
    def __init__(self, n_classes: int, in_dim: int, hidden_dims=(512, 256), p: float = 0.3):
        super().__init__()

        layers = []
        dim = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(p),
            ])
            dim = h

        layers.append(nn.Linear(dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
    



class CNN_HEAD(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 13, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(dropout * 0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x





class TRANSFORMER_HEAD(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.proj(x)  # [B, 256, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.cls(x)



class QUERY_ATTENTION_HEAD(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_dim: int,
        d_model: int = 256,
        n_queries: int = 4,
        n_heads: int = 8,
        p: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.query = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=p,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_model, d_model),
            nn.Dropout(p),
        )
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        q = self.query.expand(x.size(0), -1, -1)
        q = self.norm1(q + self.attn(q, x, x, need_weights=False)[0])
        q = self.norm2(q + self.mlp(q))
        q = q.mean(dim=1)
        return self.out(q)
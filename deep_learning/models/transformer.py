import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        mid_dim = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class PatchMerging(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(dim_out),
        )

    def forward(self, x):
        return self.reduction(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        size = (2 * window_size - 1) * (2 * window_size - 1)
        self.bias_table = nn.Parameter(torch.zeros(size, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self):
        bias = self.bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size * self.window_size, self.window_size * self.window_size, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B, C):
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rpb().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        input_resolution,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, 1, H, W))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        x = window_reverse(attn_windows, self.window_size, H, W, B, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        input_resolution,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rates=None,
        downsample=None,
        out_dim=None,
    ):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                input_resolution=input_resolution,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim, out_dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_down = self.downsample(x) if self.downsample is not None else None
        return x, x_down


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        global_pool=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.global_pool = global_pool

        self.patch_embed = ConvStem(in_chans=in_chans, embed_dim=embed_dim)

        resolution = img_size // 4
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()

        self.layers = nn.ModuleList()
        dp_offset = 0
        for i in range(len(depths)):
            layer = BasicLayer(
                dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                input_resolution=(resolution // (2 ** i), resolution // (2 ** i)),
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rates=dpr[dp_offset:dp_offset + depths[i]],
                downsample=PatchMerging if i < len(depths) - 1 else None,
                out_dim=dims[i + 1] if i < len(depths) - 1 else None,
            )
            self.layers.append(layer)
            dp_offset += depths[i]

        self.norms = nn.ModuleList([LayerNorm2d(dim) for dim in dims])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d, nn.BatchNorm2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)
            if i in self.out_indices:
                outs.append(self.norms[i](x_out))
            if x is None:
                x = x_out
        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        if self.global_pool:
            x = outs[-1]
            x = F.adaptive_avg_pool2d(x, 1)
            return x
        return outs


def transformer_backbone_tiny(img_size=224, in_chans=3, global_pool=True):
    return TransformerBackbone(
        img_size=img_size,
        in_chans=in_chans,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        global_pool=global_pool,
    )


def transformer_backbone_small(img_size=224, in_chans=3, global_pool=True):
    return TransformerBackbone(
        img_size=img_size,
        in_chans=in_chans,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        drop_path_rate=0.3,
        global_pool=global_pool,
    )


def transformer_backbone_base(img_size=224, in_chans=3, global_pool=True):
    return TransformerBackbone(
        img_size=img_size,
        in_chans=in_chans,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        mlp_ratio=4.0,
        drop_path_rate=0.5,
        global_pool=global_pool,
    )


if __name__ == "__main__":
    model = transformer_backbone_base(img_size=224, in_chans=3, global_pool=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
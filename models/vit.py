import torch
import torch.nn as nn
from einops import rearrange, repeat

from .layers import PreNorm, FeedForward, FlashAttention, pair

# ----------------------------
# Transformer
# ----------------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, H, W, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        self.H = H
        self.W = W
        self.act = nn.GELU()

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FlashAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

            self.convs.append(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            )
            self.bns.append(nn.BatchNorm2d(dim))

    def forward(self, x):
        if self.training:
            last_output = x.clone()
            nlay = len(self.layers)
            rdrop = (torch.rand((len(x), nlay), device=x.device) < 
                  torch.linspace(0, 0.4, nlay, device=x.device)).to(x.dtype)
        for i, [attn, ff] in enumerate(self.layers):
            shortcut = x[:, 1:]
            shortcut = rearrange(shortcut, 'b (h w) d -> b d h w', h=self.height, w=self.width)
            shortcut = self.gelu(shortcut)
            shortcut = self.batchnorms[i](shortcut)
            shortcut = self.convs[i](shortcut)
            shortcut = rearrange(shortcut, 'b d h w -> b (h w) d')
            cls_tokens = torch.zeros(shortcut.shape[0], 1, shortcut.shape[2], device=shortcut.device)
            shortcut = torch.cat((cls_tokens, shortcut), dim=1)
            x = attn(x) + x
            x = ff(x) + x
            x = shortcut + x
            
            # drop path
            if self.training:
                mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1)
                x = last_output * mask + (1 - mask) * x
                last_output = x.clone()

        return x


# ----------------------------
# ViT
# ----------------------------
class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        H = image_height // patch_height
        W = image_width // patch_width

        patch_dim = channels * patch_height * patch_width

        self.to_patch = nn.Sequential(
            rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, H, W, dropout
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch(img)
        b, n, _ = x.shape

        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embed[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.head(x[:, 0])
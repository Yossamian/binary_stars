import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from matplotlib.image import imread


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # x input is size (B, N+1, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # This take s and creates 3 chunks of size (B, N+1, heads*dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # Now we have 3 chunks of size (B, heads, n_patches, dim_heads)

        dots = torch.matmul(q, k.transpose(-1,
                                           -2)) * self.scale  # This works because batch dimensions are boradcasted. Results in dots shape of (B, heads, n_patches, n_patches)

        attn = self.attend(dots)  # softmax to make all in final n_patch layer sum to 1, shape is still (B, h, n, n)
        attn = self.dropout(attn)

        out = torch.matmul(attn,
                           v)  # Broadacasted matmul of (B, h, n, n) with (B, h, n, dim_heads) -> results in (b, h, n, dim_heads)
        out = rearrange(out,
                        'b h n d -> b n (h d)')  # Concatenates the resuls of all the heads to create matrix (b, n_patches, dim_output) -> where dim_output is used before linear layer
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self,
                 *,
                 image_size=900,
                 patch_size=100,
                 num_outputs=2,
                 dim=1024,
                 depth=3,
                 heads=4,
                 mlp_dim=32,
                 pool='cls',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = image_size // patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p) -> b h p', p=patch_size),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim))  # cls_token is the CLASS token - nothing to do with positional embedding
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_outputs)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img[:, :900])
        b, n, _ = x.shape  # b = batch size, n = number of patches

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # repeat the CLASS token (length dim) for each image in the batch
        x = torch.cat((cls_tokens, x),
                      dim=1)  # Add the class token to the beginning of each image in the patch -> now we sort of have N+1 patches
        x += self.pos_embedding[:,
             :(n + 1)]  # Add the positional encoding -> this is nothing but random numbers, distributed the same
        x = self.dropout(x)

        x = self.transformer(
            x)  # Input: (B, N+1, dim), Output: same dimensions. Where B=batch_size, Nis number of patches, dim is user chosen embedding size

        x = x.mean(dim=1) if self.pool == 'mean' else x[:,
                                                      0]  # When pool=='cls' this simply takes the CLASS token for each batch. Now x shape = (B, dim)

        x = self.to_latent(x)  # Basically does nothing = just identity. Shape of x is still (B, dim)
        return self.mlp_head(x)  # mlp head that uses linear layers to go from shape (B, dim) to (B, num_classes)


if __name__ == "__main__":
    image = torch.randn((945, 903))  # Shape (B, Channels, Height, Width)
    print(type(image), image.shape, image.shape[-2:])

    model = ViT(image_size=900,
                patch_size=100,
                num_outputs=2,
                dim=1024,
                depth=3,
                heads=4,
                mlp_dim=32)

    # print(model)

    output = model(image[:, :900])
    print(output.shape)

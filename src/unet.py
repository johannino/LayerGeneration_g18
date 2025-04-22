# conditional_unet_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(1, embed_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        t = t.float().unsqueeze(-1)  # [B, 1]
        x = self.linear1(t)
        x = self.act(x)
        x = self.linear2(x)
        return x  # [B, embed_dim]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding (broadcast spatially)
        t_proj = self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + t_proj

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_embed_dim=256):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim)

        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels, time_embed_dim)
        self.down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, time_embed_dim)

        # Decoder
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
)       
        self.dec1 = ResidualBlock(base_channels, base_channels, time_embed_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, t):
        B, C, H, W = x.shape

        t_emb = self.time_embedding(t)  # [B, time_embed_dim]

        # Encode
        e1 = self.enc1(x, t_emb)               # [B, base, H, W]
        e2 = self.down(e1)                     # [B, base*2, H/2, W/2]

        # Bottleneck
        m = self.mid(e2, t_emb)                # [B, base*2, H/2, W/2]

        # Decode
        u = self.up(m)                         # [B, base, H, W]
        d1 = self.dec1(u + e1, t_emb)          # skip connection

        out = self.out(d1)

        return out 

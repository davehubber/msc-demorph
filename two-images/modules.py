import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.silu = nn.SiLU()
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.silu(self.norm1(x)))
        
        emb = self.emb_layer(t)[:, :, None, None]
        h = h + emb
        
        h = self.conv2(self.silu(self.norm2(h)))
        
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x).view(B, C, -1).swapaxes(1, 2)
        
        attention_value, _ = self.mha(x_norm, x_norm, x_norm)
        
        attention_value = attention_value.swapaxes(2, 1).view(B, C, H, W)
        
        return x + attention_value


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.res_block = ResBlock(in_channels, out_channels, emb_dim)

    def forward(self, x, t):
        x = self.downsample(x)
        x = self.res_block(x, t)
        return x


class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_block = ResBlock(in_channels, out_channels, emb_dim)

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res_block(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )
        time_embed_dim = time_dim * 4

        self.inc = nn.Conv2d(c_in, 64, kernel_size=3, padding=1)
        
        self.down1 = DownResBlock(64, 128, time_embed_dim)
        self.down2 = DownResBlock(128, 256, time_embed_dim)
        
        self.sa_down = SelfAttention(256) 
        
        self.down3 = DownResBlock(256, 256, time_embed_dim)

        self.bot1 = ResBlock(256, 512, time_embed_dim)
        self.bot_attn = SelfAttention(512)
        self.bot2 = ResBlock(512, 256, time_embed_dim)

        self.up1 = UpResBlock(512, 256, time_embed_dim) 
        
        self.sa_up = SelfAttention(256)
        
        self.up2 = UpResBlock(384, 128, time_embed_dim)
        self.up3 = UpResBlock(192, 64, time_embed_dim)
        
        self.outc = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, c_out, kernel_size=3, padding=1)
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x1 = self.inc(x)
        
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa_down(x3)
        x4 = self.down3(x3, t_emb)

        x4 = self.bot1(x4, t_emb)
        x4 = self.bot_attn(x4)
        x4 = self.bot2(x4, t_emb)

        x = self.up1(x4, x3, t_emb)
        x = self.sa_up(x)
        x = self.up2(x, x2, t_emb)
        x = self.up3(x, x1, t_emb)
        
        return self.outc(x)

"""U-Net prediksi noise eps_theta(x_t, t), mengikuti arsitektur Ho et al. 2020 (Appendix B)."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.modules import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimeEmbedding,
    TimestepSequential,
    Upsample,
    zero_init,
)


class UNet(nn.Module):
    """x_t: (B, C, H, W), t: (B,) long → prediksi noise: (B, C, H, W)

    Setiap level resolusi berisi `num_res_blocks` ResBlock (+ attention jika resolusinya ada di
    `attn_resolutions`). Jalur turun menyimpan setiap aktivasi sebagai skip connection, dan jalur
    naik memakai `num_res_blocks + 1` ResBlock per level untuk mengonsumsi semuanya.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        base_channels: int,
        channel_mult: Sequence[int],
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = TimeEmbedding(base_channels, time_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        def res(cin: int, cout: int) -> ResBlock:
            return ResBlock(cin, cout, time_dim, dropout, num_groups)

        # Jalur turun
        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch, resolution = base_channels, image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [res(ch, base_channels * mult)]
                ch = base_channels * mult
                if resolution in attn_resolutions:
                    layers.append(AttentionBlock(ch, num_groups))
                self.down.append(TimestepSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(TimestepSequential(Downsample(ch)))
                skip_channels.append(ch)
                resolution //= 2

        # Bagian tengah
        self.mid = TimestepSequential(res(ch, ch), AttentionBlock(ch, num_groups), res(ch, ch))

        # Jalur naik
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_res_blocks + 1):
                layers = [res(ch + skip_channels.pop(), base_channels * mult)]
                ch = base_channels * mult
                if resolution in attn_resolutions:
                    layers.append(AttentionBlock(ch, num_groups))
                self.up.append(TimestepSequential(*layers))
            if level != 0:
                self.up.append(TimestepSequential(Upsample(ch)))
                resolution *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            zero_init(nn.Conv2d(ch, in_channels, 3, padding=1)),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        temb = self.time_embed(t)
        h = self.conv_in(x)
        skips = [h]
        for block in self.down:
            h = block(h, temb)
            skips.append(h)

        h = self.mid(h, temb)

        for block in self.up:
            if isinstance(block[0], Upsample):
                h = block(h, temb)
            else:
                h = block(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(h)

import pytest
import torch

from src.modules import AttentionBlock
from src.unet import UNet


def small_unet(image_size: int, attn_resolutions: tuple[int, ...] = (16,)) -> UNet:
    return UNet(
        image_size=image_size,
        in_channels=3,
        base_channels=32,
        channel_mult=(1, 2, 2),
        num_res_blocks=1,
        attn_resolutions=attn_resolutions,
        dropout=0.1,
        num_groups=8,
    )


@pytest.mark.parametrize("image_size", [32, 64])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_output_shape_matches_input(image_size: int, batch_size: int) -> None:
    torch.manual_seed(0)
    model = small_unet(image_size)
    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))

    assert model(x, t).shape == x.shape


def test_attention_is_placed_at_configured_resolution() -> None:
    with_attn = small_unet(32, attn_resolutions=(16,))
    without_attn = small_unet(32, attn_resolutions=())
    count = sum(isinstance(m, AttentionBlock) for m in with_attn.modules())
    # 1 di jalur turun + 2 di jalur naik (level 16×16) + 1 di tengah
    assert count == 4
    assert sum(isinstance(m, AttentionBlock) for m in without_attn.modules()) == 1


def test_output_depends_on_t() -> None:
    torch.manual_seed(0)
    model = small_unet(32).eval()
    # Output layer zero-init → output nol saat inisialisasi; acak bobotnya agar t berpengaruh.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 3, 32, 32).repeat(2, 1, 1, 1)
    out = model(x, torch.tensor([1, 900]))

    assert not torch.allclose(out[0], out[1])


def test_zero_init_output_at_start() -> None:
    model = small_unet(32)
    out = model(torch.randn(2, 3, 32, 32), torch.tensor([5, 10]))
    assert torch.all(out == 0)

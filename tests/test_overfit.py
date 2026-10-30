import torch

from src.diffusion import GaussianDiffusion
from src.unet import UNet


def test_small_model_overfits_one_batch() -> None:
    """Jika model tidak bisa overfit satu batch, masalahnya ada di model atau loss (AGENT.md §8)."""
    torch.manual_seed(0)
    model = UNet(
        image_size=16,
        in_channels=3,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        num_groups=8,
    )
    diffusion = GaussianDiffusion(timesteps=100, schedule="linear")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand(4, 3, 16, 16) * 2 - 1

    losses = []
    for _ in range(200):
        loss = diffusion.loss(model, x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first = sum(losses[:10]) / 10
    last = sum(losses[-20:]) / 20
    assert last < 0.5 * first, f"loss tidak turun signifikan: {first:.3f} → {last:.3f}"

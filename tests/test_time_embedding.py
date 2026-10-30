import torch

from src.modules import SinusoidalTimeEmbedding, TimeEmbedding


def test_sinusoidal_shape_and_range() -> None:
    emb = SinusoidalTimeEmbedding(64)(torch.tensor([0, 1, 500, 999]))
    assert emb.shape == (4, 64)
    assert emb.abs().max() <= 1.0


def test_different_t_give_different_embeddings() -> None:
    t = torch.arange(0, 1000, 37)
    emb = SinusoidalTimeEmbedding(128)(t)
    dists = torch.cdist(emb, emb)
    off_diagonal = dists[~torch.eye(len(t), dtype=torch.bool)]
    assert off_diagonal.min() > 1e-3


def test_time_embedding_mlp_shape() -> None:
    torch.manual_seed(0)
    module = TimeEmbedding(dim=32, out_dim=128)
    out = module(torch.tensor([3, 7]))
    assert out.shape == (2, 128)
    assert not torch.allclose(out[0], out[1])

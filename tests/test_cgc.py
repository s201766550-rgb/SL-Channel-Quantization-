import torch

from src.compression.cgc import CGCCompressor


def test_identical_entropy_collapses_to_one_group():
    comp = CGCCompressor(num_groups=4, bit_min=2, bit_max=8)
    entropy = torch.ones(8)
    groups, means = comp.group_channels(entropy, seed=0, round_idx=0)
    assert len(groups) == 1
    assert len(means) == 1


def test_quant_dequant_shape_and_nan_safety():
    comp = CGCCompressor(num_groups=3, bit_min=2, bit_max=8)
    x = torch.randn(2, 6, 4, 4)
    entropy = torch.rand(6) * 8.0
    c = comp.compress(x, entropy, seed=1, round_idx=0)
    y = comp.decompress(c)
    assert y.shape == x.shape
    assert torch.all(torch.isfinite(y))
    assert c.payload.min().item() >= 0
    assert c.payload.max().item() <= 255


def test_bit_clamp_range():
    comp = CGCCompressor(num_groups=2, bit_min=2, bit_max=8)
    bits = comp.allocate_bits({0: -10.0, 1: 20.0})
    assert bits[0] == 2
    assert bits[1] == 8


def test_grouping_no_empty_group_after_repair():
    comp = CGCCompressor(num_groups=4, bit_min=2, bit_max=8)
    entropy = torch.tensor([0.0, 0.1, 0.2, 9.0, 9.1, 9.2])
    groups, _ = comp.group_channels(entropy, seed=0, round_idx=0)
    assert all(len(v) > 0 for v in groups.values())

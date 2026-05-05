import torch

from src.compression.rounding import round_half_away_from_zero


def test_round_half_away_from_zero_reference_points():
    x = torch.tensor([0.5, 1.5, -0.5, -1.5], dtype=torch.float32)
    y = round_half_away_from_zero(x)
    expected = torch.tensor([1.0, 2.0, -1.0, -2.0], dtype=torch.float32)
    assert torch.equal(y, expected)

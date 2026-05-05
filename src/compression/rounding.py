import torch


def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

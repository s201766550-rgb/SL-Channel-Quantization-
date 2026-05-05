from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SplitClientModel(nn.Module):
    def __init__(self, modules: nn.ModuleList) -> None:
        super().__init__()
        self.layers = modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SplitServerModel(nn.Module):
    def __init__(self, modules: nn.ModuleList) -> None:
        super().__init__()
        self.layers = modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def create_split_resnet18(
    num_classes: int,
    split_location: str = "after_relu_first3_modules",
    pretrained: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    base = resnet18(weights=weights)
    base.fc = nn.Linear(base.fc.in_features, num_classes)

    blocks = [
        base.conv1,
        base.bn1,
        base.relu,
        base.maxpool,
        base.layer1,
        base.layer2,
        base.layer3,
        base.layer4,
        base.avgpool,
        nn.Flatten(1),
        base.fc,
    ]

    split_map = {
        "after_relu_first3_modules": 3,
        "after_maxpool": 4,
        "after_layer1": 5,
        "after_layer2": 6,
    }
    if split_location not in split_map:
        raise ValueError(f"Unsupported split_location: {split_location}")
    idx = split_map[split_location]
    client = SplitClientModel(nn.ModuleList(blocks[:idx]))
    server = SplitServerModel(nn.ModuleList(blocks[idx:]))
    return client, server

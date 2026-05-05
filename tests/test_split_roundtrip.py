import torch
import torch.nn as nn

from omegaconf import OmegaConf

from src.compression.methods import build_method_compressor
from src.split_learning.step import split_learning_parallel_concat_step


class TinyClient(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyServer(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_one_minibatch_roundtrip():
    device = torch.device("cpu")
    client = TinyClient().to(device)
    server = TinyServer().to(device)
    c_opt = torch.optim.SGD(client.parameters(), lr=1e-3)
    s_opt = torch.optim.SGD(server.parameters(), lr=1e-3)
    cfg = OmegaConf.create(
        {
            "method": {"name": "sl_acc"},
            "acii": {
                "history_window": 2,
                "eps": 1e-8,
                "log_eps": 1e-12,
                "alpha_mode": "dynamic",
                "alpha_fixed": 0.5,
            },
            "compression": {
                "num_groups": 2,
                "bit_min": 2,
                "bit_max": 8,
                "kmeans_n_init": 10,
                "kmeans_max_iter": 300,
            },
        }
    )
    method_compressor = build_method_compressor(cfg)

    batch_by_client = [
        (0, torch.randn(2, 3, 8, 8), torch.tensor([0, 1], dtype=torch.long)),
        (1, torch.randn(2, 3, 8, 8), torch.tensor([1, 2], dtype=torch.long)),
    ]

    metrics = split_learning_parallel_concat_step(
        client_model=client,
        server_model=server,
        client_optimizer=c_opt,
        server_optimizer=s_opt,
        batch_by_client=batch_by_client,
        method_compressor=method_compressor,
        round_idx=0,
        total_rounds=2,
        seed=0,
    )
    assert metrics.active_client_count == 2
    assert metrics.activation_upload_payload_bits > 0
    assert metrics.gradient_download_payload_bits > 0

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.compression.rounding import round_half_away_from_zero
from src.compression.types import CompressedTensor
from src.metrics.communication import metadata_bits, payload_bits, runtime_bytes_of_tensors


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 2:
        return x.unsqueeze(-1).unsqueeze(-1)
    raise ValueError(f"Expected [B,C,H,W] or [B,C], got {tuple(x.shape)}")


class CGCCompressor:
    def __init__(
        self,
        num_groups: int = 4,
        bit_min: int = 2,
        bit_max: int = 8,
        eps: float = 1e-8,
        n_init: int = 10,
        max_iter: int = 300,
    ) -> None:
        self.num_groups = num_groups
        self.bit_min = bit_min
        self.bit_max = bit_max
        self.eps = eps
        self.n_init = n_init
        self.max_iter = max_iter

    def _repair_empty_clusters(
        self, entropy: np.ndarray, labels: np.ndarray, effective_groups: int
    ) -> np.ndarray:
        while True:
            counts = np.bincount(labels, minlength=effective_groups)
            empty = np.where(counts == 0)[0]
            if len(empty) == 0:
                return labels
            largest = int(np.argmax(counts))
            largest_idx = np.where(labels == largest)[0]
            if len(largest_idx) <= 1:
                return labels
            vals = entropy[largest_idx]
            median = float(np.median(vals))
            left = largest_idx[vals <= median]
            right = largest_idx[vals > median]
            if len(right) == 0:
                right = largest_idx[len(largest_idx) // 2 :]
                left = largest_idx[: len(largest_idx) // 2]
            target_empty = int(empty[0])
            labels[right] = target_empty
            if len(np.unique(labels)) >= min(effective_groups, len(np.unique(entropy))):
                return labels

    def group_channels(
        self, entropy: torch.Tensor, seed: int, round_idx: int
    ) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
        e_np = entropy.detach().cpu().numpy().astype(np.float64)
        c = len(e_np)
        unique_vals = len(np.unique(e_np))
        effective = int(min(self.num_groups, c, unique_vals))
        if effective <= 1:
            return {0: list(range(c))}, {0: float(np.mean(e_np))}

        kmeans = KMeans(
            n_clusters=effective,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=int(seed + round_idx),
        )
        labels = kmeans.fit_predict(e_np.reshape(-1, 1))
        labels = self._repair_empty_clusters(e_np, labels, effective)

        groups: Dict[int, List[int]] = {}
        group_mean_entropy: Dict[int, float] = {}
        for gid in range(effective):
            idx = np.where(labels == gid)[0].tolist()
            if len(idx) == 0:
                continue
            groups[gid] = idx
            group_mean_entropy[gid] = float(np.mean(e_np[idx]))

        # Deterministic ordering: re-index groups by mean entropy
        ordered = sorted(group_mean_entropy.items(), key=lambda kv: kv[1])
        remap = {old_gid: new_gid for new_gid, (old_gid, _) in enumerate(ordered)}
        groups_out: Dict[int, List[int]] = {}
        means_out: Dict[int, float] = {}
        for old_gid, mean_val in ordered:
            new_gid = remap[old_gid]
            groups_out[new_gid] = groups[old_gid]
            means_out[new_gid] = mean_val
        return groups_out, means_out

    def allocate_bits(self, group_mean_entropy: Dict[int, float]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for gid, m in group_mean_entropy.items():
            bits = int(np.floor(m))
            bits = max(self.bit_min, min(self.bit_max, bits))
            out[gid] = bits
        return out

    def compress(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor,
        seed: int,
        round_idx: int,
    ) -> CompressedTensor:
        x4 = _ensure_4d(x)
        payload = torch.zeros_like(x4, dtype=torch.uint8)

        group_to_channels, group_mean_entropy = self.group_channels(entropy, seed, round_idx)
        bits = self.allocate_bits(group_mean_entropy)
        group_x_min: Dict[int, float] = {}
        group_x_max: Dict[int, float] = {}
        group_num_values: Dict[int, int] = {}

        for gid, channels in group_to_channels.items():
            group_vals = x4[:, channels, :, :]
            x_min = float(group_vals.min().item())
            x_max = float(group_vals.max().item())
            group_x_min[gid] = x_min
            group_x_max[gid] = x_max
            group_num_values[gid] = int(group_vals.numel())

            if x_max == x_min:
                payload[:, channels, :, :] = 0
                continue

            levels = (2 ** bits[gid]) - 1
            scale = levels / (x_max - x_min + self.eps)
            qf = (group_vals - x_min) * scale
            q = round_half_away_from_zero(qf)
            q = torch.clamp(q, 0, levels).to(torch.uint8)
            payload[:, channels, :, :] = q

        shape4 = list(x4.shape)
        logical_payload = payload_bits(group_num_values, bits)
        logical_meta = metadata_bits(shape4, group_to_channels, bits)
        runtime_bytes = runtime_bytes_of_tensors([payload])

        return CompressedTensor(
            payload=payload,
            original_shape=tuple(x4.shape),
            group_to_channels=group_to_channels,
            group_bit_widths=bits,
            group_x_min=group_x_min,
            group_x_max=group_x_max,
            storage_dtype=str(payload.dtype),
            logical_payload_bits=logical_payload,
            logical_metadata_bits=logical_meta,
            runtime_tensor_bytes=runtime_bytes,
        )

    def decompress(self, c: CompressedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        out = torch.zeros(c.original_shape, dtype=dtype, device=c.payload.device)
        for gid, channels in c.group_to_channels.items():
            x_min = c.group_x_min[gid]
            x_max = c.group_x_max[gid]
            q = c.payload[:, channels, :, :].to(dtype)
            if x_max == x_min:
                out[:, channels, :, :] = x_min
                continue
            levels = (2 ** c.group_bit_widths[gid]) - 1
            out[:, channels, :, :] = q / levels * (x_max - x_min) + x_min
        return out

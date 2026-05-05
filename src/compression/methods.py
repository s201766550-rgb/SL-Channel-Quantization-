from typing import Dict, List

import torch

from src.compression.acii import ACII
from src.compression.cgc import CGCCompressor
from src.compression.rounding import round_half_away_from_zero
from src.compression.types import CompressedTensor
from src.metrics.communication import runtime_bytes_of_tensors


class BaseMethodCompressor:
    def compress(
        self,
        x: torch.Tensor,
        client_id: int,
        direction: str,
        round_idx: int,
        total_rounds: int,
        seed: int,
    ) -> CompressedTensor:
        raise NotImplementedError

    def decompress(self, c: CompressedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        raise NotImplementedError

    def finalize_round(self) -> None:
        return None

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, payload: Dict) -> None:
        _ = payload


class SLACCCompressor(BaseMethodCompressor):
    def __init__(self, cfg) -> None:
        self.acii = ACII(
            history_window=int(cfg.acii.history_window),
            eps=float(cfg.acii.eps),
            log_eps=float(cfg.acii.log_eps),
            importance_metric=str(cfg.acii.importance_metric),
        )
        self.cgc = CGCCompressor(
            num_groups=int(cfg.compression.num_groups),
            bit_min=int(cfg.compression.bit_min),
            bit_max=int(cfg.compression.bit_max),
            n_init=int(cfg.compression.kmeans_n_init),
            max_iter=int(cfg.compression.kmeans_max_iter),
        )
        self.alpha_mode = str(cfg.acii.alpha_mode)
        self.alpha_fixed = float(cfg.acii.alpha_fixed)

    def compress(
        self,
        x: torch.Tensor,
        client_id: int,
        direction: str,
        round_idx: int,
        total_rounds: int,
        seed: int,
    ) -> CompressedTensor:
        scores, _ = self.acii.score_tensor(
            x=x,
            client_id=client_id,
            direction=direction,
            round_idx=round_idx,
            total_rounds=total_rounds,
            alpha_mode=self.alpha_mode,
            alpha_fixed=self.alpha_fixed,
        )
        return self.cgc.compress(x=x, entropy=scores, seed=seed, round_idx=round_idx)

    def decompress(self, c: CompressedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.cgc.decompress(c, dtype=dtype)

    def finalize_round(self) -> None:
        self.acii.finalize_round()

    def state_dict(self) -> Dict:
        return {"acii": self.acii.state_dict()}

    def load_state_dict(self, payload: Dict) -> None:
        if "acii" in payload:
            self.acii.load_state_dict(payload["acii"])


def _linear_quant(x: torch.Tensor, bits: int, eps: float = 1e-8):
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_max == x_min:
        q = torch.zeros_like(x, dtype=torch.uint8)
        return q, x_min, x_max
    levels = (2**bits) - 1
    qf = (x - x_min) / (x_max - x_min + eps) * levels
    q = torch.clamp(round_half_away_from_zero(qf), 0, levels).to(torch.uint8)
    return q, x_min, x_max


def _linear_dequant(q: torch.Tensor, x_min: float, x_max: float, bits: int, dtype: torch.dtype):
    if x_max == x_min:
        return torch.full_like(q, fill_value=x_min, dtype=dtype)
    levels = (2**bits) - 1
    return q.to(dtype) / levels * (x_max - x_min) + x_min


class SplitFCCompressor(BaseMethodCompressor):
    def __init__(self, keep_ratio: float = 0.5, bits: int = 8):
        self.keep_ratio = keep_ratio
        self.bits = bits

    def compress(self, x, client_id, direction, round_idx, total_rounds, seed):
        _ = (client_id, direction, round_idx, total_rounds, seed)
        x4 = x if x.dim() == 4 else x.unsqueeze(-1).unsqueeze(-1)
        c = x4.shape[1]
        std = x4.flatten(2).std(dim=2).mean(dim=0)
        keep = max(1, int(c * self.keep_ratio))
        keep_idx = torch.topk(std, k=keep).indices
        mask = torch.zeros(c, dtype=torch.bool, device=x4.device)
        mask[keep_idx] = True
        retained = x4[:, mask, :, :]
        q, x_min, x_max = _linear_quant(retained, self.bits)
        logical_payload_bits = int(retained.numel() * self.bits)
        logical_meta_bits = 4 * 32 + 32 + 32 + int(keep * 16)
        runtime_bytes = runtime_bytes_of_tensors([q, keep_idx.to(torch.int32)])
        return CompressedTensor(
            payload=q,
            original_shape=tuple(x4.shape),
            group_to_channels={0: keep_idx.cpu().tolist()},
            group_bit_widths={0: self.bits},
            group_x_min={0: x_min},
            group_x_max={0: x_max},
            storage_dtype=str(q.dtype),
            logical_payload_bits=logical_payload_bits,
            logical_metadata_bits=logical_meta_bits,
            runtime_tensor_bytes=runtime_bytes,
            extra_metadata={"kept_channels": keep_idx.cpu().tolist()},
        )

    def decompress(self, c: CompressedTensor, dtype=torch.float32):
        out = torch.zeros(c.original_shape, device=c.payload.device, dtype=dtype)
        idx = c.group_to_channels[0]
        rec = _linear_dequant(
            c.payload, c.group_x_min[0], c.group_x_max[0], c.group_bit_widths[0], dtype
        )
        out[:, idx, :, :] = rec
        return out


class PowerQuantCompressor(BaseMethodCompressor):
    def __init__(self, bits: int = 8, exponent: float = 2.0):
        self.bits = bits
        self.exponent = exponent

    def compress(self, x, client_id, direction, round_idx, total_rounds, seed):
        _ = (client_id, direction, round_idx, total_rounds, seed)
        x4 = x if x.dim() == 4 else x.unsqueeze(-1).unsqueeze(-1)
        transformed = torch.sign(x4) * torch.pow(torch.abs(x4), 1.0 / self.exponent)
        q, x_min, x_max = _linear_quant(transformed, self.bits)
        logical_payload_bits = int(x4.numel() * self.bits)
        logical_meta_bits = 4 * 32 + 32 + 32
        runtime_bytes = runtime_bytes_of_tensors([q])
        return CompressedTensor(
            payload=q,
            original_shape=tuple(x4.shape),
            group_to_channels={0: list(range(x4.shape[1]))},
            group_bit_widths={0: self.bits},
            group_x_min={0: x_min},
            group_x_max={0: x_max},
            storage_dtype=str(q.dtype),
            logical_payload_bits=logical_payload_bits,
            logical_metadata_bits=logical_meta_bits,
            runtime_tensor_bytes=runtime_bytes,
            extra_metadata={"exponent": self.exponent},
        )

    def decompress(self, c: CompressedTensor, dtype=torch.float32):
        transformed = _linear_dequant(
            c.payload, c.group_x_min[0], c.group_x_max[0], c.group_bit_widths[0], dtype
        )
        exp = float(c.extra_metadata["exponent"])
        return torch.sign(transformed) * torch.pow(torch.abs(transformed), exp)


class RandTopKCompressor(BaseMethodCompressor):
    def __init__(self, topk_ratio: float = 0.1, random_extra_ratio: float = 0.01):
        self.topk_ratio = topk_ratio
        self.random_extra_ratio = random_extra_ratio

    def compress(self, x, client_id, direction, round_idx, total_rounds, seed):
        _ = (client_id, direction, total_rounds)
        flat = x.flatten()
        n = flat.numel()
        k = max(1, int(n * self.topk_ratio))
        r = max(0, int(n * self.random_extra_ratio))
        topk_idx = torch.topk(flat.abs(), k=k).indices
        mask = torch.ones(n, dtype=torch.bool, device=x.device)
        mask[topk_idx] = False
        remaining = torch.arange(n, device=x.device)[mask]
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed + round_idx + client_id))
        if r > 0 and remaining.numel() > 0:
            ridx = remaining[torch.randperm(remaining.numel(), generator=g)[: min(r, remaining.numel())]]
            idx = torch.unique(torch.cat([topk_idx, ridx]))
        else:
            idx = topk_idx
        vals = flat[idx].to(torch.float32)
        logical_payload_bits = int(vals.numel() * 32)
        # include indices and shape in metadata
        logical_meta_bits = int(4 * 32 + idx.numel() * 32)
        runtime_bytes = runtime_bytes_of_tensors([vals, idx.to(torch.int32)])
        return CompressedTensor(
            payload=vals,
            original_shape=tuple(x.shape),
            group_to_channels={0: []},
            group_bit_widths={0: 32},
            group_x_min={0: 0.0},
            group_x_max={0: 0.0},
            storage_dtype=str(vals.dtype),
            logical_payload_bits=logical_payload_bits,
            logical_metadata_bits=logical_meta_bits,
            runtime_tensor_bytes=runtime_bytes,
            extra_metadata={"flat_indices": idx.cpu().tolist()},
        )

    def decompress(self, c: CompressedTensor, dtype=torch.float32):
        out = torch.zeros(int(torch.tensor(c.original_shape).prod().item()), dtype=dtype, device=c.payload.device)
        idx = torch.tensor(c.extra_metadata["flat_indices"], device=out.device, dtype=torch.long)
        out[idx] = c.payload.to(dtype)
        return out.reshape(c.original_shape)


class EasyQuantCompressor(PowerQuantCompressor):
    def __init__(self, bits: int = 8, calibration_batches: int = 32):
        super().__init__(bits=bits, exponent=1.0)
        self.calibration_batches = calibration_batches
        self._seen = 0
        self._frozen_min = None
        self._frozen_max = None

    def compress(self, x, client_id, direction, round_idx, total_rounds, seed):
        _ = (client_id, direction, round_idx, total_rounds, seed)
        x4 = x if x.dim() == 4 else x.unsqueeze(-1).unsqueeze(-1)
        if self._seen < self.calibration_batches:
            curr_min = float(x4.min().item())
            curr_max = float(x4.max().item())
            self._frozen_min = curr_min if self._frozen_min is None else min(self._frozen_min, curr_min)
            self._frozen_max = curr_max if self._frozen_max is None else max(self._frozen_max, curr_max)
            self._seen += 1
        x_min = self._frozen_min if self._frozen_min is not None else float(x4.min().item())
        x_max = self._frozen_max if self._frozen_max is not None else float(x4.max().item())
        if x_max == x_min:
            q = torch.zeros_like(x4, dtype=torch.uint8)
        else:
            levels = (2**self.bits) - 1
            qf = (x4 - x_min) / (x_max - x_min + 1e-8) * levels
            q = torch.clamp(round_half_away_from_zero(qf), 0, levels).to(torch.uint8)
        return CompressedTensor(
            payload=q,
            original_shape=tuple(x4.shape),
            group_to_channels={0: list(range(x4.shape[1]))},
            group_bit_widths={0: self.bits},
            group_x_min={0: x_min},
            group_x_max={0: x_max},
            storage_dtype=str(q.dtype),
            logical_payload_bits=int(x4.numel() * self.bits),
            logical_metadata_bits=4 * 32 + 32 + 32,
            runtime_tensor_bytes=runtime_bytes_of_tensors([q]),
        )


def build_method_compressor(cfg) -> BaseMethodCompressor:
    method = str(cfg.method.name).lower()
    if method in ["sl_acc", "acii_only", "cgc_only", "random_channel", "std_channel"]:
        if method == "random_channel":
            cfg.acii.importance_metric = "random"
        elif method == "std_channel":
            cfg.acii.importance_metric = "std"
        return SLACCCompressor(cfg)
    if method == "splitfc":
        return SplitFCCompressor(keep_ratio=float(cfg.method.std_keep_ratio), bits=8)
    if method == "powerquant_sl":
        return PowerQuantCompressor(
            bits=int(getattr(cfg.method, "quantization_bits", 8)),
            exponent=float(cfg.method.powerquant_exponent),
        )
    if method == "randtopk_sl":
        return RandTopKCompressor(
            topk_ratio=float(cfg.method.topk_ratio),
            random_extra_ratio=float(cfg.method.random_extra_ratio),
        )
    if method == "easyquant":
        return EasyQuantCompressor(
            bits=int(getattr(cfg.method, "quantization_bits", 8)),
            calibration_batches=int(cfg.method.easyquant_calibration_batches),
        )
    if method == "powerquant":
        return PowerQuantCompressor(bits=8, exponent=float(cfg.method.powerquant_exponent))
    raise ValueError(f"Unsupported method: {cfg.method.name}")

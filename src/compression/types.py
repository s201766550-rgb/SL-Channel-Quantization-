from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CompressedTensor:
    payload: torch.Tensor
    original_shape: Tuple[int, ...]
    group_to_channels: Dict[int, List[int]]
    group_bit_widths: Dict[int, int]
    group_x_min: Dict[int, float]
    group_x_max: Dict[int, float]
    storage_dtype: str
    logical_payload_bits: int
    logical_metadata_bits: int
    runtime_tensor_bytes: int
    extra_metadata: Optional[Dict[str, Any]] = None

    @property
    def logical_total_bits(self) -> int:
        return self.logical_payload_bits + self.logical_metadata_bits

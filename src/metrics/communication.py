from typing import Dict, List

import torch


def payload_bits(group_num_values: Dict[int, int], group_bit_widths: Dict[int, int]) -> int:
    total = 0
    for gid, num_values in group_num_values.items():
        total += int(num_values) * int(group_bit_widths[gid])
    return int(total)


def metadata_bits(
    original_shape_len4: List[int],
    group_to_channels: Dict[int, List[int]],
    group_bit_widths: Dict[int, int],
) -> int:
    _ = group_bit_widths
    # shape: 4 x 32 bits
    total = 4 * 32
    # number of groups
    total += 32
    for gid, channels in group_to_channels.items():
        _ = gid
        # bit width: 8, min:32, max:32, channel count:32
        total += 8 + 32 + 32 + 32
        # channel indices: 16 bits each
        total += len(channels) * 16
    return int(total)


def runtime_bytes_of_tensors(tensors: List[torch.Tensor]) -> int:
    total = 0
    for t in tensors:
        total += t.numel() * t.element_size()
    return int(total)


def fp32_uncompressed_payload_bits(num_elements: int) -> int:
    return int(num_elements) * 32

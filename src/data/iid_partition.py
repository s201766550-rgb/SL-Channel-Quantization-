from typing import Dict, List

import numpy as np


def iid_partition(num_samples: int, num_clients: int, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    base = num_samples // num_clients
    rem = num_samples % num_clients
    out: Dict[int, List[int]] = {}
    start = 0
    for cid in range(num_clients):
        extra = 1 if cid < rem else 0
        end = start + base + extra
        out[cid] = indices[start:end].tolist()
        start = end
    return out

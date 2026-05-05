from typing import Dict, List, Sequence, Tuple

import numpy as np


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    beta: float,
    seed: int,
    batch_size: int,
    max_resample_attempts: int = 100,
) -> Tuple[Dict[int, List[int]], bool]:
    labels_np = np.asarray(labels)
    classes = np.unique(labels_np)
    rng = np.random.default_rng(seed)
    best_partition = None
    best_min_size = -1

    for _ in range(max_resample_attempts):
        client_indices = {cid: [] for cid in range(num_clients)}
        for cls in classes:
            cls_idx = np.where(labels_np == cls)[0]
            rng.shuffle(cls_idx)
            props = rng.dirichlet(np.full(num_clients, beta))
            split_points = (np.cumsum(props) * len(cls_idx)).astype(int)[:-1]
            shards = np.split(cls_idx, split_points)
            for cid, shard in enumerate(shards):
                client_indices[cid].extend(shard.tolist())
        min_size = min(len(v) for v in client_indices.values())
        if min_size > best_min_size:
            best_partition = client_indices
            best_min_size = min_size
        if min_size >= batch_size:
            return client_indices, False

    assert best_partition is not None
    return best_partition, True

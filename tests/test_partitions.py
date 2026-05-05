import numpy as np

from src.data.dirichlet_partition import dirichlet_partition
from src.data.iid_partition import iid_partition


def test_iid_partition_even_with_remainder():
    parts = iid_partition(num_samples=11, num_clients=3, seed=0)
    sizes = [len(parts[i]) for i in range(3)]
    assert sizes == [4, 4, 3]
    merged = sorted(sum((parts[i] for i in range(3)), []))
    assert merged == list(range(11))


def test_dirichlet_partition_returns_all_indices():
    labels = np.array([i % 4 for i in range(100)])
    parts, warned = dirichlet_partition(
        labels=labels,
        num_clients=5,
        beta=0.5,
        seed=0,
        batch_size=4,
        max_resample_attempts=10,
    )
    merged = sorted(sum(parts.values(), []))
    assert merged == list(range(100))
    assert isinstance(warned, bool)

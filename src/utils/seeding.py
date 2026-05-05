import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


def make_torch_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    if device and device.startswith("cuda"):
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen

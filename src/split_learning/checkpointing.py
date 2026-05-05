import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _rng_state() -> Dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    data = dict(payload)
    data["rng_states"] = _rng_state()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, p)


def load_checkpoint(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if "rng_states" in data:
        _set_rng_state(data["rng_states"])
    return data

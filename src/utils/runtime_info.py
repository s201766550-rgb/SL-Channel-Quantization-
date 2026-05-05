import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torchvision


def _git_commit(project_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_runtime_info(project_root: str) -> Dict[str, Any]:
    gpu_name = "none"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "platform": platform.platform(),
        "git_commit": _git_commit(project_root),
    }


def write_runtime_info(project_root: str, out_file: str) -> None:
    payload = collect_runtime_info(project_root)
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HAM_LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str, tfm: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tfm = tfm
        self.label_map = HAM_LABEL_MAP

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        x = self.tfm(image)
        y = self.label_map[row["dx"]]
        return x, y


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _mnist_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _normalize_ham_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    # Supported formats:
    # 1) Canonical HAM metadata: columns include image_id and dx
    # 2) One-hot labels (e.g., GroundTruth.csv): image + class columns
    col_lookup = {str(c).strip().lower(): c for c in frame.columns}

    image_id_col = col_lookup.get("image_id")
    dx_col = col_lookup.get("dx")
    if image_id_col is not None and dx_col is not None:
        out = frame[[image_id_col, dx_col]].copy()
        out.columns = ["image_id", "dx"]
        out["image_id"] = (
            out["image_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.(jpg|jpeg|png)$", "", case=False, regex=True)
        )
        out["dx"] = out["dx"].astype(str).str.strip().str.lower()
        return out

    image_col = col_lookup.get("image")
    onehot_cols = {k: v for k, v in col_lookup.items() if k in HAM_LABEL_MAP}
    if image_col is not None and onehot_cols:
        class_keys = list(onehot_cols.keys())
        class_values = (
            frame[[onehot_cols[k] for k in class_keys]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        winner_idx = class_values.argmax(axis=1)
        out = pd.DataFrame(
            {
                "image_id": (
                    frame[image_col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\.(jpg|jpeg|png)$", "", case=False, regex=True)
                ),
                "dx": [class_keys[i] for i in winner_idx],
            }
        )
        return out

    raise ValueError(
        "Unsupported HAM metadata format. Expected either (image_id, dx) columns or "
        "(image + one-hot class columns like MEL/NV/BCC/AKIEC/BKL/DF/VASC). "
        f"Found columns: {list(frame.columns)}"
    )


def load_dataset(cfg) -> Tuple[Dataset, Dataset, Dict[str, bool]]:
    name = cfg.dataset.name.lower()
    image_size = int(cfg.dataset.image_size)
    download_root = cfg.dataset.download_root
    allow_download = bool(cfg.dataset.allow_download)
    meta = {"dirichlet_resample_warning": False}

    if name == "mnist":
        train_ds = datasets.MNIST(
            root=download_root,
            train=True,
            download=allow_download,
            transform=_mnist_transform(image_size),
        )
        test_ds = datasets.MNIST(
            root=download_root,
            train=False,
            download=allow_download,
            transform=_mnist_transform(image_size),
        )
        return train_ds, test_ds, meta
    if name == "cifar100":
        tfm = build_transforms(image_size)
        train_ds = datasets.CIFAR100(
            root=download_root, train=True, download=allow_download, transform=tfm
        )
        test_ds = datasets.CIFAR100(
            root=download_root, train=False, download=allow_download, transform=tfm
        )
        return train_ds, test_ds, meta
    if name == "ham10000":
        image_dir = Path(cfg.dataset.ham10000.image_dir)
        metadata_csv = Path(cfg.dataset.ham10000.metadata_csv)
        if not image_dir.exists() or not metadata_csv.exists():
            raise FileNotFoundError(
                "HAM10000 paths are missing. Set dataset.ham10000.image_dir and "
                "dataset.ham10000.metadata_csv to valid Kaggle input paths."
            )
        frame = _normalize_ham_metadata(pd.read_csv(metadata_csv))
        train_frame, test_frame = train_test_split(
            frame,
            test_size=float(cfg.dataset.test_ratio),
            random_state=int(cfg.seed),
            stratify=frame["dx"],
            shuffle=True,
        )
        tfm = build_transforms(image_size)
        train_ds = HAM10000Dataset(train_frame, str(image_dir), tfm)
        test_ds = HAM10000Dataset(test_frame, str(image_dir), tfm)
        return train_ds, test_ds, meta
    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")


def dataset_labels(ds: Dataset) -> List[int]:
    if hasattr(ds, "targets"):
        targets = getattr(ds, "targets")
        if isinstance(targets, torch.Tensor):
            return targets.cpu().tolist()
        return list(targets)
    if isinstance(ds, HAM10000Dataset):
        return [ds.label_map[v] for v in ds.frame["dx"].tolist()]
    raise ValueError("Cannot extract labels from dataset type")

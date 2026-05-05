from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import shutil

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.compression.methods import build_method_compressor
from src.data.dataset_loader import dataset_labels, load_dataset
from src.data.dirichlet_partition import dirichlet_partition
from src.data.iid_partition import iid_partition
from src.models.resnet18_split import create_split_resnet18
from src.split_learning.checkpointing import load_checkpoint, save_checkpoint
from src.split_learning.step import split_learning_parallel_concat_step
from src.utils.logging_io import MetricLogger, write_json
from src.utils.runtime_info import write_runtime_info
from src.utils.seeding import make_torch_generator, make_worker_init_fn


def _resolve_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_loaders(cfg, train_ds):
    num_clients = int(cfg.client.num_clients)
    batch_size = int(cfg.training.batch_size)
    seed = int(cfg.seed)
    if bool(cfg.distribution.iid):
        parts = iid_partition(len(train_ds), num_clients, seed)
        warning = False
    else:
        labels = dataset_labels(train_ds)
        parts, warning = dirichlet_partition(
            labels=labels,
            num_clients=num_clients,
            beta=float(cfg.distribution.dirichlet_beta),
            seed=seed,
            batch_size=batch_size,
            max_resample_attempts=int(cfg.distribution.max_resample_attempts),
        )
    loaders = {}
    for cid, idx in parts.items():
        subset = Subset(train_ds, idx)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(cfg.runtime.num_workers),
            pin_memory=bool(cfg.runtime.pin_memory),
            persistent_workers=bool(cfg.runtime.persistent_workers),
            worker_init_fn=make_worker_init_fn(seed),
            generator=make_torch_generator(seed),
        )
        loaders[cid] = loader
    return loaders, parts, warning


@torch.no_grad()
def evaluate(client_model, server_model, test_loader, device: torch.device) -> float:
    client_model.eval()
    server_model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits = server_model(client_model(x))
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    client_model.train()
    server_model.train()
    return 100.0 * correct / max(total, 1)


def train(cfg) -> Dict:
    save_dir = Path(str(cfg.paths.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(str(cfg.paths.checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_runtime_info(str(cfg.paths.project_root), str(save_dir / "runtime_info.json"))
    train_ds, test_ds, meta = load_dataset(cfg)
    loaders_by_client, partitions, warning = _build_loaders(cfg, train_ds)
    if warning:
        meta["dirichlet_resample_warning"] = True

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=bool(cfg.runtime.pin_memory),
    )

    device = _resolve_device(str(cfg.runtime.device))
    client_model, server_model = create_split_resnet18(
        num_classes=int(cfg.dataset.num_classes),
        split_location=str(cfg.model.split_location),
        pretrained=bool(cfg.model.pretrained),
    )
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    c_opt = torch.optim.SGD(
        client_model.parameters(),
        lr=float(cfg.optimizer.lr),
        momentum=float(cfg.optimizer.momentum),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    s_opt = torch.optim.SGD(
        server_model.parameters(),
        lr=float(cfg.optimizer.lr),
        momentum=float(cfg.optimizer.momentum),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    method_compressor = build_method_compressor(cfg)

    start_round = 0
    best_acc = 0.0
    if bool(cfg.training.resume) and cfg.training.resume_path:
        ckpt = load_checkpoint(str(cfg.training.resume_path))
        client_model.load_state_dict(ckpt["client_model_state"])
        server_model.load_state_dict(ckpt["server_model_state"])
        c_opt.load_state_dict(ckpt["client_optimizer_state"])
        s_opt.load_state_dict(ckpt["server_optimizer_state"])
        method_compressor.load_state_dict(ckpt.get("method_compressor_state", {}))
        start_round = int(ckpt["round_idx"]) + 1
        best_acc = float(ckpt.get("best_accuracy", 0.0))

    logger = MetricLogger(str(save_dir))
    wandb_run = None
    if bool(cfg.tracking.wandb_enabled):
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=str(cfg.tracking.wandb_project),
                entity=cfg.tracking.wandb_entity,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception:
            wandb_run = None
    total_rounds = int(cfg.training.rounds)
    global_step = 0
    for round_idx in range(start_round, total_rounds):
        iters = {cid: iter(loader) for cid, loader in loaders_by_client.items()}
        active = set(iters.keys())
        while active:
            batch_by_client = []
            exhausted = []
            use_parallel_concat = bool(cfg.client.client_parallel) and str(cfg.server.server_batching_mode) == "parallel_concat"
            if use_parallel_concat:
                for cid in list(active):
                    try:
                        x, y = next(iters[cid])
                        batch_by_client.append((cid, x.to(device), y.to(device)))
                    except StopIteration:
                        exhausted.append(cid)
            else:
                cid = sorted(list(active))[0]
                try:
                    x, y = next(iters[cid])
                    batch_by_client.append((cid, x.to(device), y.to(device)))
                except StopIteration:
                    exhausted.append(cid)
            for cid in exhausted:
                active.remove(cid)
            if not batch_by_client:
                continue

            try:
                metrics = split_learning_parallel_concat_step(
                    client_model=client_model,
                    server_model=server_model,
                    client_optimizer=c_opt,
                    server_optimizer=s_opt,
                    batch_by_client=batch_by_client,
                    method_compressor=method_compressor,
                    round_idx=round_idx,
                    total_rounds=total_rounds,
                    seed=int(cfg.seed),
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "CUDA OOM encountered. Try lowering training.batch_size "
                        "(e.g., 64 or 32) or using runtime.gradient_accumulation_steps."
                    ) from exc
                raise
            row = asdict(metrics)
            row["round_idx"] = round_idx
            row["step_idx"] = global_step
            row["total_payload_bits"] = (
                row["activation_upload_payload_bits"] + row["gradient_download_payload_bits"]
            )
            row["total_bits_with_metadata"] = (
                row["activation_upload_total_bits_with_metadata"]
                + row["gradient_download_total_bits_with_metadata"]
            )
            row["total_runtime_bytes"] = (
                row["activation_upload_runtime_bytes"] + row["gradient_download_runtime_bytes"]
            )
            logger.log(row)
            if wandb_run is not None:
                wandb_run.log(row)
            global_step += 1

        method_compressor.finalize_round()

        if (round_idx + 1) % int(cfg.training.eval_interval) == 0:
            acc = evaluate(client_model, server_model, test_loader, device)
            logger.log({"round_idx": round_idx, "metric": "test_accuracy", "value": acc})
            if wandb_run is not None:
                wandb_run.log({"round_idx": round_idx, "test_accuracy": acc})
            best_acc = max(best_acc, acc)

        if (round_idx + 1) % int(cfg.training.checkpoint_interval_rounds) == 0:
            ckpt_payload = {
                "client_model_state": client_model.state_dict(),
                "server_model_state": server_model.state_dict(),
                "client_optimizer_state": c_opt.state_dict(),
                "server_optimizer_state": s_opt.state_dict(),
                "round_idx": round_idx,
                "seed": int(cfg.seed),
                "partition_indices": partitions,
                "method_compressor_state": method_compressor.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "best_accuracy": best_acc,
            }
            if bool(cfg.training.save_latest_checkpoint):
                save_checkpoint(str(checkpoint_dir / "latest.pt"), ckpt_payload)
            if bool(cfg.training.save_best_checkpoint):
                save_checkpoint(str(checkpoint_dir / "best.pt"), ckpt_payload)

    logger.close()
    if wandb_run is not None:
        wandb_run.finish()
    summary = {
        "best_accuracy": best_acc,
        "dirichlet_resample_warning": meta["dirichlet_resample_warning"],
    }
    write_json(str(save_dir / "final_summary.json"), summary)
    OmegaConf.save(config=cfg, f=str(save_dir / "config.yaml"))
    assumptions_src = Path(str(cfg.paths.project_root)) / "REPRODUCTION_ASSUMPTIONS.md"
    if assumptions_src.exists():
        shutil.copyfile(assumptions_src, save_dir / "reproduction_assumptions.md")
    return summary

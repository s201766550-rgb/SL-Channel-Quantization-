from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.compression.methods import BaseMethodCompressor


@dataclass
class StepMetrics:
    activation_upload_payload_bits: int
    activation_upload_total_bits_with_metadata: int
    activation_upload_runtime_bytes: int
    gradient_download_payload_bits: int
    gradient_download_total_bits_with_metadata: int
    gradient_download_runtime_bytes: int
    active_client_count: int
    loss: float


def split_learning_accumulate_step(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    client_optimizer: torch.optim.Optimizer,
    server_optimizer: torch.optim.Optimizer,
    batch_by_client: List[Tuple[int, torch.Tensor, torch.Tensor]],
    method_compressor: BaseMethodCompressor,
    round_idx: int,
    total_rounds: int,
    seed: int,
) -> StepMetrics:
    client_optimizer.zero_grad()
    server_optimizer.zero_grad()

    act_payload = act_total = act_runtime = 0
    grad_payload = grad_total = grad_runtime = 0
    total_loss = 0.0

    num_clients = len(batch_by_client)

    for client_id, x, y in batch_by_client:
        a = client_model(x)

        c_act = method_compressor.compress(
            x=a.detach(),
            client_id=client_id,
            direction="activation_upload",
            round_idx=round_idx,
            total_rounds=total_rounds,
            seed=seed,
        )
        act_payload += c_act.logical_payload_bits
        act_total += c_act.logical_total_bits
        act_runtime += c_act.runtime_tensor_bytes

        a_tilde = method_compressor.decompress(c_act, dtype=a.dtype).detach().requires_grad_(True)
        
        logits = server_model(a_tilde)
        
        # Divide loss by num_clients for gradient accumulation
        loss = F.cross_entropy(logits, y) / num_clients
        loss.backward()
        
        # Keep original loss scale for reporting
        total_loss += loss.item() * num_clients

        grad_a = a_tilde.grad
        c_grad = method_compressor.compress(
            x=grad_a.detach(),
            client_id=client_id,
            direction="gradient_download",
            round_idx=round_idx,
            total_rounds=total_rounds,
            seed=seed,
        )
        grad_payload += c_grad.logical_payload_bits
        grad_total += c_grad.logical_total_bits
        grad_runtime += c_grad.runtime_tensor_bytes
        
        grad_tilde = method_compressor.decompress(c_grad, dtype=a.dtype)
        a.backward(grad_tilde)

    server_optimizer.step()
    client_optimizer.step()

    return StepMetrics(
        activation_upload_payload_bits=act_payload,
        activation_upload_total_bits_with_metadata=act_total,
        activation_upload_runtime_bytes=act_runtime,
        gradient_download_payload_bits=grad_payload,
        gradient_download_total_bits_with_metadata=grad_total,
        gradient_download_runtime_bytes=grad_runtime,
        active_client_count=num_clients,
        loss=total_loss,
    )

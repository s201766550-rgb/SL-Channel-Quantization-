from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import torch
import torch.nn.functional as F


class ACII:
    def __init__(
        self,
        history_window: int = 5,
        eps: float = 1e-8,
        log_eps: float = 1e-12,
        importance_metric: str = "entropy",
    ) -> None:
        self.history_window = history_window
        self.eps = eps
        self.log_eps = log_eps
        self.importance_metric = importance_metric
        self._round_history: DefaultDict[Tuple[int, str], List[torch.Tensor]] = defaultdict(list)
        self._current_round_batches: DefaultDict[Tuple[int, str], List[torch.Tensor]] = defaultdict(list)

    def _ensure_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x.unsqueeze(-1).unsqueeze(-1)
        raise ValueError(f"ACII expects [B,C,H,W] or [B,C], got shape {tuple(x.shape)}")

    def channel_entropy(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(x)
        b, c, h, w = x.shape
        flat = x.permute(1, 0, 2, 3).reshape(c, b * h * w)

        ent = torch.zeros(c, device=x.device, dtype=x.dtype)
        mins = flat.min(dim=1).values
        maxs = flat.max(dim=1).values
        constant_mask = maxs == mins

        if (~constant_mask).any():
            valid = flat[~constant_mask]
            valid_min = mins[~constant_mask].unsqueeze(1)
            valid_max = maxs[~constant_mask].unsqueeze(1)
            normalized = (valid - valid_min) / (valid_max - valid_min + self.eps)
            probs = F.softmax(normalized, dim=1)
            ent_valid = -(probs * torch.log(probs + self.log_eps)).sum(dim=1)
            ent[~constant_mask] = ent_valid
        # Constant channels remain zero by definition.
        return ent

    def channel_std(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(x)
        b, c, h, w = x.shape
        flat = x.permute(1, 0, 2, 3).reshape(c, b * h * w)
        return flat.std(dim=1)

    def channel_importance(self, x: torch.Tensor) -> torch.Tensor:
        metric = self.importance_metric.lower()
        if metric == "entropy":
            return self.channel_entropy(x)
        if metric == "std":
            return self.channel_std(x)
        if metric == "random":
            x4 = self._ensure_4d(x)
            return torch.rand(x4.shape[1], device=x4.device, dtype=x4.dtype)
        raise ValueError(f"Unsupported importance_metric: {self.importance_metric}")

    def score_tensor(
        self,
        x: torch.Tensor,
        client_id: int,
        direction: str,
        round_idx: int,
        total_rounds: int,
        alpha_mode: str = "dynamic",
        alpha_fixed: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (client_id, direction)
        current = self.channel_importance(x)
        self._current_round_batches[key].append(current.detach().cpu())

        t = round_idx + 1
        if alpha_mode == "dynamic":
            alpha = float(t) / float(total_rounds)
        else:
            alpha = float(alpha_fixed)

        history = self._round_history[key]
        if t <= self.history_window:
            history_items = history + [current.detach().cpu()]
        else:
            history_items = history[-self.history_window :]
            if not history_items:
                history_items = [current.detach().cpu()]
        history_mean = torch.stack(history_items, dim=0).mean(dim=0).to(current.device, current.dtype)

        combined = (1.0 - alpha) * current + alpha * history_mean
        return combined, current

    def finalize_round(self) -> None:
        for key, batch_vals in self._current_round_batches.items():
            round_mean = torch.stack(batch_vals, dim=0).mean(dim=0)
            self._round_history[key].append(round_mean)
        self._current_round_batches.clear()

    def state_dict(self) -> Dict[str, Dict[str, List[List[float]]]]:
        out: Dict[str, Dict[str, List[List[float]]]] = {}
        for (cid, direction), values in self._round_history.items():
            client_key = str(cid)
            out.setdefault(client_key, {})
            out[client_key][direction] = [v.tolist() for v in values]
        return out

    def load_state_dict(self, payload: Dict[str, Dict[str, List[List[float]]]]) -> None:
        self._round_history.clear()
        for client_key, mapping in payload.items():
            cid = int(client_key)
            for direction, values in mapping.items():
                self._round_history[(cid, direction)] = [torch.tensor(v, dtype=torch.float32) for v in values]

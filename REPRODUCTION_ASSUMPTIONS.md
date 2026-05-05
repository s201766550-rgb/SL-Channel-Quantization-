# SL-ACC Reproduction Assumptions

This file captures implementation locks used for the zero-gap reproduction.

## Locked Defaults

- Optimizer: SGD (`lr=1e-4`, `momentum=0.9`, `weight_decay=5e-4`)
- Rounds: `200`
- Batch size: `128`
- Clients: `5`
- Split: `after_relu_first3_modules`
- Quantization bits: `[2, 8]`
- Groups: `4`
- History window: `5`
- Seed: `0`
- Default profile tracker: local CSV/JSONL.

## ACII

- Entropy uses per-channel min-max normalization followed by softmax.
- Constant channel (`max == min`) entropy is `0.0`.
- History behavior:
  - `t <= k`: history includes current entropy.
  - `t > k`: history uses only previous `k` rounds.
- `alpha` default is dynamic `t / T`.

## CGC

- Grouping: deterministic 1D K-means++.
- Empty clusters repaired by splitting largest non-empty group around median entropy.
- Identical entropy values collapse into one group.
- Bit allocation: `floor(group_mean_entropy)` clamped to `[bit_min, bit_max]`.

## Communication Accounting

- Report all three quantities:
  - `paper_payload_bits`
  - `logical_total_bits` (payload + metadata)
  - `runtime_tensor_bytes`
- Metadata bit accounting:
  - shape: `4 * 32`
  - group count: `32`
  - per-group:
    - bit width: `8`
    - min/max: `32 + 32`
    - channel count: `32`
    - channel indices: `16 * n_channels`

## Activation/Gradient Paths

- Activation upload and gradient download are compressed independently.
- Separate ACII history buffers per direction.
- Server uses:
  - `A_tilde = dequantized_activation.detach().requires_grad_(True)`
  - server backward derives `grad_A = A_tilde.grad`.
- Client backward uses dequantized gradient payload.

## Kaggle Execution

- Default Kaggle execution is single-process logical parallelism (`parallel_concat`).
- No required DDP/multi-node/multi-GPU for default runs.
- Writes must go under `/kaggle/working`; never write to `/kaggle/input`.

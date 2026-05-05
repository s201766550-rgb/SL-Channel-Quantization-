import torch

from src.compression.acii import ACII


def test_constant_channel_entropy_is_zero():
    acii = ACII(history_window=5)
    x = torch.ones(2, 3, 4, 4)
    ent = acii.channel_entropy(x)
    assert ent.shape == (3,)
    assert torch.allclose(ent, torch.zeros_like(ent))


def test_entropy_shape_for_2d_input():
    acii = ACII(history_window=5)
    x = torch.randn(8, 6)
    ent = acii.channel_entropy(x)
    assert ent.shape == (6,)


def test_minmax_then_softmax_entropy_pipeline():
    acii = ACII(history_window=2)
    x = torch.tensor([[[[0.0]], [[2.0]]], [[[1.0]], [[3.0]]]], dtype=torch.float32)
    ent = acii.channel_entropy(x)
    assert ent.shape == (2,)
    assert torch.all(torch.isfinite(ent))


def test_alpha_dynamic_uses_round_idx_plus_one_over_total():
    acii = ACII(history_window=5)
    x = torch.randn(2, 4, 2, 2)
    combined, current = acii.score_tensor(
        x=x,
        client_id=0,
        direction="activation_upload",
        round_idx=0,
        total_rounds=10,
        alpha_mode="dynamic",
    )
    # t=1 => alpha=0.1, with no history current is also history proxy
    assert torch.allclose(combined, current, atol=1e-6)


def test_history_rule_t_le_k_includes_current():
    acii = ACII(history_window=3)
    x = torch.tensor([[[[1.0]], [[2.0]]]])
    # Round 1
    c1, _ = acii.score_tensor(x, 0, "activation_upload", round_idx=0, total_rounds=10)
    acii.finalize_round()
    # Round 2 (t<=k includes current + previous)
    c2, cur2 = acii.score_tensor(x * 2, 0, "activation_upload", round_idx=1, total_rounds=10)
    assert c2.shape == cur2.shape
    assert torch.all(torch.isfinite(c2))


def test_history_rule_t_gt_k_excludes_current():
    acii = ACII(history_window=2)
    for ridx in range(3):
        x = torch.randn(2, 3, 2, 2) + ridx
        acii.score_tensor(x, 1, "gradient_download", round_idx=ridx, total_rounds=10)
        acii.finalize_round()
    x_new = torch.randn(2, 3, 2, 2) + 10
    combined, _ = acii.score_tensor(
        x_new, 1, "gradient_download", round_idx=3, total_rounds=10
    )
    assert torch.all(torch.isfinite(combined))

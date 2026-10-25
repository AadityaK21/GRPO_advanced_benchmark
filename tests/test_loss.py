"""Checks for src/train/loss.py."""
import math

import pytest
import torch

from src.train.loss import grpo_loss

MASK = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])   # completion 1 has 2 tokens, completion 2 has 3
ADV = torch.tensor([1.0, -2.0])


def _one_token(ratio, advantage, **kwargs):
    """Loss and d(loss)/d(logp) for a single token whose probability ratio new/old is `ratio`."""
    logps = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss = grpo_loss(logps, torch.zeros(1, 1), torch.tensor([advantage]), torch.ones(1, 1), **kwargs)
    loss.backward()
    return loss.item(), logps.grad.item()


# -------------------------------------------------------- aggregation


@pytest.mark.parametrize(
    "loss_type, max_len, want",
    [
        ("grpo", None, (-1.0 + 2.0) / 2),           # mean per completion, then over completions
        ("bnpo", None, (-1.0 * 2 + 2.0 * 3) / 5),    # mean over all 5 tokens
        ("dapo", None, (-1.0 * 2 + 2.0 * 3) / 5),    # same as bnpo within one batch
        ("dr_grpo", 8, (-1.0 * 2 + 2.0 * 3) / (2 * 8)),  # sum / (B * max_completion_length)
    ],
)
def test_on_policy_loss_value(loss_type, max_len, want):
    logps = torch.full((2, 3), -1.0)
    loss = grpo_loss(logps, None, ADV, MASK, loss_type=loss_type, max_completion_length=max_len)
    assert math.isclose(loss.item(), want, rel_tol=1e-6), f"at ratio 1 each token's loss is -A; then {loss_type}'s average"


@pytest.mark.parametrize(
    "loss_type, short, long",
    [
        ("grpo", -1 / (2 * 2), -1 / (2 * 4)),   # each completion counts equally, so short ones' tokens weigh more
        ("bnpo", -1 / 6, -1 / 6),               # each token counts equally
        ("dr_grpo", -1 / 8, -1 / 8),            # each token counts equally, over a fixed denominator
    ],
)
def test_on_policy_gradient_weights_each_token_by_loss_type(loss_type, short, long):
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    logps = torch.zeros(2, 4, requires_grad=True)
    grpo_loss(logps, None, torch.ones(2), mask, loss_type=loss_type, max_completion_length=4).backward()
    want = torch.tensor([[short, short, 0.0, 0.0], [long] * 4])
    torch.testing.assert_close(
        logps.grad, want, msg="this weighting is exactly what Dr. GRPO changes; masked tokens get zero gradient"
    )


def test_old_logps_none_means_on_policy():
    logps = torch.randn(2, 3, generator=torch.Generator().manual_seed(0)).requires_grad_()
    a = grpo_loss(logps, None, ADV, MASK)
    a.backward()
    grad_none = logps.grad.clone()
    logps.grad = None
    b = grpo_loss(logps, logps.detach().clone(), ADV, MASK)
    b.backward()
    assert math.isclose(a.item(), b.item(), rel_tol=1e-6)
    torch.testing.assert_close(grad_none, logps.grad, msg="None means old = logps.detach(): ratio 1, gradient intact")


def test_masked_tokens_do_not_matter():
    logps = torch.full((2, 3), -1.0)
    base = grpo_loss(logps, None, ADV, MASK).item()
    logps[0, 2] = -50.0   # change only the masked token
    old = logps.detach().clone()
    old[0, 2] = 7.0
    assert math.isclose(grpo_loss(logps, old, ADV, MASK).item(), base, rel_tol=1e-6)


# ------------------------------------------------------------ clipping


def test_hand_computed_clipped_values():
    assert math.isclose(_one_token(1.5, 1.0)[0], -1.2, rel_tol=1e-6), "-min(1.5 * 1, 1.2 * 1)"
    assert math.isclose(_one_token(0.5, 1.0)[0], -0.5, rel_tol=1e-6), "-min(0.5 * 1, 0.8 * 1)"


@pytest.mark.parametrize("ratio, advantage", [(1.5, 1.0), (0.5, -1.0)])
def test_clipping_stops_the_gradient_once_the_update_went_far_enough(ratio, advantage):
    assert _one_token(ratio, advantage)[1] == 0.0


@pytest.mark.parametrize("ratio, advantage", [(1.5, -1.0), (0.5, 1.0)])
def test_clipping_never_hides_a_move_in_the_wrong_direction(ratio, advantage):
    assert _one_token(ratio, advantage)[1] != 0.0, "min() keeps the unclipped term when the policy moved the wrong way"


def test_upper_and_lower_clip_ranges_are_separate():
    assert _one_token(1.25, 1.0, eps_high=0.2)[1] == 0.0
    assert _one_token(1.25, 1.0, eps_high=0.28)[1] != 0.0, "the upper bound is 1 + eps_high (DAPO's clip-higher)"
    assert _one_token(0.75, -1.0, eps_low=0.2)[1] == 0.0
    assert _one_token(0.75, -1.0, eps_low=0.3)[1] != 0.0, "the lower bound is 1 - eps_low"


def test_ratio_and_clipping_are_per_token():
    logps = torch.tensor([[math.log(1.5), 0.0]], requires_grad=True)
    grpo_loss(logps, torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2)).backward()
    assert logps.grad[0, 0] == 0.0 and logps.grad[0, 1] != 0.0, "only the token past the clip range loses its gradient"


@pytest.mark.parametrize("advantage, direction", [(1.0, 1), (-1.0, -1)])
def test_gradient_has_the_right_sign(advantage, direction):
    grad = _one_token(1.0, advantage)[1]
    assert -grad * direction > 0, "minimising the loss must raise the log-prob of a positive-advantage token"


def test_no_gradient_into_old_logps_ref_logps_or_advantages():
    logps = torch.randn(2, 3, generator=torch.Generator().manual_seed(1)).requires_grad_()
    old = (logps.detach() + 0.1).requires_grad_()
    ref = (logps.detach() - 0.1).requires_grad_()
    adv = ADV.clone().requires_grad_()
    grpo_loss(logps, old, adv, MASK, ref_logps=ref, beta=0.1).backward()
    assert logps.grad is not None
    assert old.grad is None and ref.grad is None and adv.grad is None, "detach them: they are constants"


# ------------------------------------------------------------------ KL


def test_kl_is_zero_when_the_policy_equals_the_reference():
    logps = torch.randn(2, 3, generator=torch.Generator().manual_seed(2))
    with_kl = grpo_loss(logps, None, ADV, MASK, ref_logps=logps.clone(), beta=0.5)
    without = grpo_loss(logps, None, ADV, MASK)
    assert math.isclose(with_kl.item(), without.item(), rel_tol=1e-6, abs_tol=1e-7)


def test_kl_hand_computed_value():
    logps = torch.tensor([[-1.0]])
    loss = grpo_loss(logps, None, torch.zeros(1), torch.ones(1, 1), ref_logps=torch.tensor([[-2.0]]), beta=0.1)
    d = -2.0 - (-1.0)
    assert math.isclose(loss.item(), 0.1 * (math.exp(d) - d - 1), rel_tol=1e-6), "k3: exp(ref - logp) - (ref - logp) - 1"


@pytest.mark.parametrize("logp, ref, sign", [(-1.0, -2.0, 1), (-2.0, -1.0, -1)])
def test_kl_pulls_the_policy_toward_the_reference(logp, ref, sign):
    logps = torch.tensor([[logp]], requires_grad=True)
    grpo_loss(logps, None, torch.zeros(1), torch.ones(1, 1), ref_logps=torch.tensor([[ref]]), beta=0.1).backward()
    assert logps.grad.item() * sign > 0, "a gradient step must move logp toward ref_logp"


def test_unknown_loss_type_is_an_error():
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 1), None, torch.zeros(1), torch.ones(1, 1), loss_type="grpoo")


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dapo", "dr_grpo"])
def test_a_row_with_no_counted_tokens_is_not_in_the_loss(loss_type):
    """A keep-mask drops a completion by zeroing its row of the mask (MC-GRPO). The loss must be the
    loss of the other rows alone: the dropped row is not even in a denominator."""
    g = torch.Generator().manual_seed(3)
    logps = -torch.rand(6, 5, generator=g)
    adv = torch.randn(6, generator=g)
    mask = torch.ones(6, 5)
    mask[0, 3:] = 0
    keep = torch.tensor([True, True, False, True, True, True])
    dropped = grpo_loss(logps, None, adv, mask * keep.unsqueeze(1), loss_type=loss_type, max_completion_length=5)
    alone = grpo_loss(logps[keep], None, adv[keep], mask[keep], loss_type=loss_type, max_completion_length=5)
    torch.testing.assert_close(dropped, alone)

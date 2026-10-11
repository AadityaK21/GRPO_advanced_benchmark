"""Your GRPO pieces against TRL's GRPOTrainer, on the same batch.

The plan's rule: this file must pass before any run counts.

TRL samples one batch with its own code. Then each side computes, with its own code: the completion
mask, the advantages, the per-token log-probs, and the loss with its gradients. The loss is checked
twice: at the first update, where the ratio is exactly 1, and at a later update, after the weights
have moved, where clipping happens. Every TRL loss option is set explicitly in _trainer().

Pinned: trl==1.15.0 and transformers==5.19.0. This file calls two private TRL methods,
_generate_and_score_completions and _compute_loss, which can change in any release.

TRL 1.15 computes log-probs with a Triton kernel, which needs a GPU on Linux. Where that kernel
can run, this file uses it: run it on Kaggle or the HPC, because that is the real check, and
test_uses_trls_gpu_kernel then passes. Anywhere else (a CPU, or Windows), it swaps the kernel for
the same maths in plain PyTorch (_TorchLogProbs below) so the rest of TRL's code can run, and
test_uses_trls_gpu_kernel is skipped. That run compares your code with TRL's loss code, not with
TRL's kernel.
"""
import copy
import tempfile

import pytest
import torch

pytest.importorskip("trl")
import trl.trainer.utils as trl_utils  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from src.advantages.grpo import group_advantages  # noqa: E402
from src.train.logprobs import per_token_logps  # noqa: E402
from src.train.loss import grpo_loss  # noqa: E402
from src.train.sample import completion_mask  # noqa: E402
from tiny import EOS, make_model, make_tokenizer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TRL_KERNEL = DEVICE == "cuda" and trl_utils._ChunkedLogProbFunction is not None   # None without Triton
PROMPTS = ["t1 t2 t3", "t9"]   # the second one gets left-padded
G = 8                          # completions per prompt
MAX_LEN = 6                    # max completion length
REWARDS = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,   # fixed by position, so neither group is uniform
           0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

# Tolerances, written down as the plan asks. On CPU the two sides agree exactly; on a GPU,
# TRL's chunked kernel adds numbers in a different order, so allow float32 rounding.
LOGP_TOL = dict(rtol=1e-4, atol=1e-5)
LOSS_TOL = dict(rtol=1e-4, atol=1e-6)
GRAD_TOL = 1e-3   # largest |your grad - TRL's grad|, as a fraction of TRL's largest grad


class _TorchLogProbs:
    """Plain-PyTorch version of TRL's ChunkedLogProbFunction (trl/kernels/chunked_logprob.py), for CPUs."""

    @staticmethod
    def apply(hidden, weight, bias, labels, temperature, chunk_size, softcap=None, logit_scale=1.0,
              outputs=("log_probs",)):
        z = hidden @ weight.t()
        if bias is not None:
            z = z + bias
        z = z * logit_scale
        if softcap is not None:
            z = softcap * torch.tanh(z / softcap)
        z = z / temperature
        log_probs = z.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(z, dim=-1)
        entropy = None
        if "entropy" in outputs:
            entropy = -(torch.softmax(z, dim=-1) * torch.log_softmax(z, dim=-1)).sum(dim=-1)
        return log_probs, entropy, None, None, None


@pytest.fixture(autouse=True)
def _exact_float32(monkeypatch):
    if not USE_TRL_KERNEL:
        monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", _TorchLogProbs)
    if DEVICE == "cuda":  # no TF32 shortcuts in matrix products
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


def test_uses_trls_gpu_kernel():
    if not USE_TRL_KERNEL:
        pytest.skip("no GPU with Triton here, so these tests used the plain-PyTorch stand-in for TRL's "
                    "log-prob kernel. The real check is on Kaggle or the HPC.")
    assert trl_utils._ChunkedLogProbFunction is not _TorchLogProbs


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """TRL rebuilds its reference model from the model's folder, so the tiny model needs one."""
    path = tmp_path_factory.mktemp("tiny-qwen2")
    make_model(seed=0).save_pretrained(path)
    return str(path)


def _load(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, attn_implementation="eager", dtype=torch.float32)
    return model.to(DEVICE).eval()


def _trainer(model_dir, loss_type="grpo", beta=0.0, temperature=1.0, eps_high=0.2, scale="group"):
    args = GRPOConfig(
        output_dir=tempfile.mkdtemp(), report_to=[], logging_steps=1000,
        use_cpu=DEVICE == "cpu", bf16=False, fp16=False, gradient_checkpointing=False, disable_dropout=True,
        per_device_train_batch_size=len(PROMPTS) * G, num_generations=G, max_completion_length=MAX_LEN,
        num_iterations=2,   # makes TRL store the sampling-time log-probs as old_per_token_logps
        temperature=temperature, top_k=0, top_p=1.0, repetition_penalty=1.0,
        loss_type=loss_type, beta=beta, epsilon=0.2, epsilon_high=eps_high, scale_rewards=scale,
        importance_sampling_level="token", use_bias_correction_kl=False, delta=None,
        top_entropy_quantile=1.0, mask_truncated_completions=False, entropy_coef=0.0,
    )
    return GRPOTrainer(
        model=_load(model_dir), args=args, processing_class=make_tokenizer(),
        reward_funcs=lambda completions, **kwargs: REWARDS[: len(completions)],
        train_dataset=Dataset.from_dict({"prompt": PROMPTS}),
    )


def _sample(trainer):
    trainer.model.eval()
    torch.manual_seed(0)
    return trainer._generate_and_score_completions([{"prompt": p} for p in PROMPTS for _ in range(G)])


def _model_inputs(batch):
    input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
    attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
    return input_ids, attention_mask, batch["completion_ids"].size(1)


def _nudge(model):
    """The same fixed change to the weights of both models, standing in for an optimizer step."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.add_(0.05 * torch.randn(p.shape, generator=g).to(p.device))


@pytest.mark.parametrize("scale", ["group", "none"])
def test_mask_and_advantages_match_trl(model_dir, scale):
    batch = _sample(_trainer(model_dir, scale=scale))
    mine = completion_mask(batch["completion_ids"], EOS)
    assert torch.equal(mine.long().cpu(), batch["completion_mask"].long().cpu())
    torch.testing.assert_close(group_advantages(REWARDS, G, scale=scale).cpu(), batch["advantages"].float().cpu(),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_log_probs_match_trl(model_dir, temperature):
    batch = _sample(_trainer(model_dir, temperature=temperature))
    input_ids, attention_mask, k = _model_inputs(batch)
    with torch.no_grad():
        mine = per_token_logps(_load(model_dir), input_ids, attention_mask, k, temperature)
    counted = batch["completion_mask"].bool()   # TRL leaves 0 where the mask is 0
    torch.testing.assert_close(mine[counted], batch["old_per_token_logps"][counted], **LOGP_TOL)


@pytest.mark.parametrize("update, eps_high", [("first", 0.2), ("later", 0.2), ("later", 0.28)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo"])
def test_loss_and_gradients_match_trl(model_dir, loss_type, beta, temperature, update, eps_high):
    trainer = _trainer(model_dir, loss_type, beta, temperature, eps_high)
    batch = _sample(trainer)
    trl_model = trainer.model

    # Your side: your mask, advantages and log-probs, from a fresh copy of the same weights.
    mine = _load(model_dir)
    ref_model = copy.deepcopy(mine)
    input_ids, attention_mask, k = _model_inputs(batch)
    mask = completion_mask(batch["completion_ids"], EOS)
    advantages = group_advantages(REWARDS, G).to(DEVICE)
    with torch.no_grad():
        old = per_token_logps(mine, input_ids, attention_mask, k, temperature)
        ref = per_token_logps(ref_model, input_ids, attention_mask, k, temperature) if beta else None

    inputs = dict(batch)
    if update == "first":
        inputs.pop("old_per_token_logps")   # TRL then uses its current log-probs, detached: ratio exactly 1
        old = None
    else:
        _nudge(trl_model)
        _nudge(mine)

    trl_loss = trainer._compute_loss(trl_model, inputs)
    trl_loss.backward()
    logps = per_token_logps(mine, input_ids, attention_mask, k, temperature)
    my_loss = grpo_loss(logps, old, advantages, mask, ref_logps=ref, beta=beta, eps_low=0.2, eps_high=eps_high,
                        loss_type=loss_type, max_completion_length=MAX_LEN)
    my_loss.backward()

    torch.testing.assert_close(my_loss.detach(), trl_loss.detach(), **LOSS_TOL)
    trl_grads = {n: p.grad for n, p in trl_model.named_parameters() if p.grad is not None}
    my_grads = {n: p.grad for n, p in mine.named_parameters() if p.grad is not None}
    assert my_grads.keys() == trl_grads.keys(), "both losses should reach the same weights"
    largest = max(g.abs().max().item() for g in trl_grads.values())
    worst, name = max(((my_grads[n] - g).abs().max().item(), n) for n, g in trl_grads.items())
    assert worst <= GRAD_TOL * largest, f"gradients differ by {worst:.2e} in {name} (TRL's largest: {largest:.2e})"

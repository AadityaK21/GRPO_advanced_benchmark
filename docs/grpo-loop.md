# Your GRPO loop (weeks 5–7)

You write the code in `src/`. The tests in `tests/` tell you when each piece is right. They check properties, not one particular implementation, so any correct GRPO passes. Finish the REINFORCE and PPO warm-ups first.

The tests use a tiny Qwen2 model (2 layers, a 64-token vocabulary) built from a config, so they run offline on a laptop CPU in under a minute. It is the same model code as Qwen2.5-0.5B, only smaller. `tests/conftest.py` switches the Hugging Face Hub off while tests run, so a test can never download anything by accident.

## Setup

One virtual environment for the whole repo, warm-ups included. In PowerShell:

```
cd C:\GRPO
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch gymnasium numpy pytest transformers==5.19.0 trl==1.15.0 datasets peft pyyaml scikit-learn matplotlib scipy
```

Python 3.11 or newer (TRL 1.15 needs 3.11). If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. Activate the environment each time you open a new terminal.

On Windows, `pip install torch` gives the CPU build. That is all the tests need: they run on the CPU, and the GPU work happens on Kaggle or the HPC.

These are the versions I tested with (torch 2.14.1). Write them into `pyproject.toml` and the shared-loop table in `implementation-notes.md`. TRL's internals change between releases, so the TRL check is only valid for the pinned version.

## Run

```
pytest                         # everything in tests/
pytest tests/test_loss.py      # one file
pytest -k mask                 # only tests with "mask" in their name
```

Right now 110 tests fail with `NotImplementedError` and 1 is skipped. That's the starting point. The skipped one, `test_uses_trls_gpu_kernel`, only runs where TRL's GPU kernel can run (next section). Without TRL installed, all of `test_against_trl.py` is skipped and 58 tests remain.

## Work in this order

| Step | You write | Tests that check it | What it teaches |
|---|---|---|---|
| 1 | `src/reward/gsm8k.py` | `test_reward.py` | what counts as a correct GSM8K answer |
| 2 | `src/advantages/grpo.py` | `test_advantages.py` | the group baseline, and why a group where every answer is right (or every one wrong) gives no gradient |
| 3 | `completion_mask` in `src/train/sample.py` | `test_sample.py -k mask` | which completion tokens are trained on |
| 4 | `src/train/logprobs.py` | `test_logprobs.py` | the one-token shift, left padding, and where the temperature goes |
| 5 | `src/train/loss.py` | `test_loss.py` | the clipped ratio, the KL term, and the token averaging that Dr. GRPO and DAPO change |
| 6 | `sample_group` in `src/train/sample.py` | the rest of `test_sample.py` | sampling from exactly the distribution your log-probs describe |
| 7 | `src/train/loop.py` | `test_loop.py` | one full step: old log-probs, several updates per batch, the KL to a frozen reference |
| 8 | nothing new | `test_against_trl.py`, on a GPU | your pieces compute what TRL computes, on the same batch |
| 9 | your training script | short runs against TRL | your loop learns like TRL's |

The last test in `test_loop.py` trains the tiny model on a toy task with your loop. It should go from about 0.14 reward to above 0.8 within 100 steps, in a few seconds.

Read while you work: DeepSeekMath (2402.03300) for GRPO itself (steps 2 and 5), Dr. GRPO (2503.20783) for `scale="none"` and `loss_type="dr_grpo"`, LoRA (2106.09685) before the real model, and 2605.25252 before the short runs, to see what accuracy to expect.

## The TRL check

`test_against_trl.py` lets TRL sample one batch with its own code. Then both sides compute, each with its own code: the completion mask, the advantages, the per-token log-probs, and the loss with its gradients. The loss is checked at the first update, where the ratio is exactly 1, and at a later update, after the weights have moved, where clipping happens. That covers 4 loss types, with and without KL, at two temperatures, with symmetric and asymmetric clipping. The tolerances are written at the top of the file, as the plan asks.

**Run it on a GPU.** TRL 1.15's GRPO trainer always computes log-probs with a Triton kernel, which needs a GPU. A comment in TRL's source says Triton ships with PyTorch on Linux only, so this can't run for real on your Windows laptop. Where the kernel can't run, the test swaps it for the same maths in plain PyTorch, so the rest of TRL's code still runs, and `test_uses_trls_gpu_kernel` is skipped. That run is a useful first check. The real one is on Kaggle or the HPC, where `test_uses_trls_gpu_kernel` passes instead of being skipped.

On Kaggle: a new notebook with the GPU accelerator and Internet switched on (Kaggle asks you to verify a phone number once). Put a read-only GitHub token for the repo in the notebook's Secrets as `GITHUB_TOKEN`, then:

```
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://AadityaK21:{token}@github.com/AadityaK21/GRPO_advanced_benchmark.git
%cd GRPO_advanced_benchmark
!pip install -q transformers==5.19.0 trl==1.15.0 pytest
!pytest tests/test_against_trl.py
```

TRL refuses to start when one process sees two GPUs, as with Kaggle's "GPU T4 x2". `tests/conftest.py` makes the tests see only the first one.

**What I could and couldn't check.** I ran every test against my own private solution on a Linux CPU. In the CPU version of the TRL check, the two sides agreed to within float32 rounding (about 1e-7). I couldn't run a GPU here, so the GPU path is untested. If TRL's kernel fails on Kaggle's T4, try the HPC. If the test fails on a GPU by a little, say a gradient off by 2e-3 of the largest, find out why before you loosen a tolerance, and write the reason in `implementation-notes.md`.

## Then the real model

Your training script loads `Qwen/Qwen2.5-0.5B-Instruct`, formats each GSM8K question with the chat template (`tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)`), and calls `grpo_step` in a loop. Three things the tiny model can't show you:

- **Its generation_config.json sets temperature 0.7, top_k 20, top_p 0.8 and repetition_penalty 1.1.** `generate()` uses any of these you don't override. `test_sample.py` checks that your sampler overrides them.
- **It stops on either of two EOS tokens**, 151645 and 151643. Mask with both.
- **With LoRA, set `lora_dropout=0`** for the checks. Dropout makes every log-prob random, so nothing would match exactly. The reference model is the same model with the adapter switched off (`with model.disable_adapter():` in peft), so you don't need a second copy in memory.

## Then short runs against TRL (week 7)

Same model, same prompts, same settings, 3 or more seeds each, about 100 steps. Plot reward per step for both. The seed bands should overlap. They won't match exactly, because the samples differ.

Settings must match, and TRL's defaults are easy to miss. These are TRL 1.15.0's:

| TRL setting | Default | What your loop must do |
|---|---|---|
| `learning_rate` | 1e-6 | same |
| `lr_scheduler_type`, `warmup_steps` | `"linear"`, 0 | decay the learning rate linearly to 0 over the run, or set TRL to `"constant"` |
| `max_grad_norm` | 1.0 | `max_grad_norm=1.0` in `GRPOSettings`: clip the gradient norm before each update |
| `optim`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay` | AdamW, 0.9, 0.999, 1e-8, 0.0 | the same AdamW |
| `per_device_train_batch_size`, `num_generations` | 8, 8 | one prompt with 8 completions per step |
| `num_iterations` | 1 | `num_updates` |
| `loss_type`, `beta` | `"dapo"`, 0.0 | set both explicitly on both sides |
| `epsilon`, `epsilon_high` | 0.2, None (means 0.2) | `eps_low`, `eps_high` |
| `scale_rewards` | `"group"` | `scale` |
| `use_bias_correction_kl` | True | multiplies the KL term by the ratio; set it False, as the test does |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | 1.0, 0, 1.0, 1.0 | your sampler |
| `max_completion_length` | 512 | `max_new_tokens` |
| `mask_truncated_completions` | False | completions cut off at the limit still count |
| `shuffle_dataset`, `seed` | True, 42 | easiest: give both the same pre-shuffled prompts and turn shuffling off |
| `gradient_checkpointing` | True | saves memory, doesn't change the maths |

## What the tests catch

I put each of these 52 bugs, one at a time, into a working solution of my own. Each one failed at least one test, without needing TRL:

- **Reward:** the first number instead of the last; `####` ignored; commas not stripped; `18.00` not normalised to `18`.
- **Advantages:** the std without Bessel's correction; no `1e-4`, so a uniform group gives NaN; groups taken as strided rows; `scale="none"` still dividing; the batch mean instead of the group mean.
- **Mask:** the EOS token itself masked out; only the first of several EOS ids; masking from the last EOS instead of the first.
- **Sampling:** prompts padded on the right; prompts tiled instead of grouped; `do_sample` not set, so decoding is greedy; the temperature not passed; the checkpoint's `top_k`, `top_p` or `repetition_penalty` left on, whether you call `generate()` with arguments or with a `GenerationConfig` object.
- **Log-probs:** no one-token shift; no temperature; the temperature applied after the log-softmax; the attention mask not passed; the output detached.
- **Loss:** `max` instead of `min`; `eps_low` used for the upper bound; the ratio as a difference instead of the exp of a difference; the whole loss with the wrong sign; `old_logps=None` without a detach; advantages, old log-probs or reference log-probs not detached; the KL with its sign flipped; the KL without `beta`; `"grpo"` averaged over tokens instead of completions; `"dr_grpo"` divided by the token count, or by the batch's length instead of `max_completion_length`; masked tokens summed in.
- **Loop:** answers tiled instead of grouped; old log-probs recomputed at every update; the KL measured against the model itself; rewards flipped; no `optimizer.zero_grad()`; the reported reward wrong; any of the temperature, `scale`, `loss_type`, `beta`, `max_new_tokens`, the clip range or the reference model's temperature not reaching the piece that uses it.

## What they can't catch

Every bug I tried was caught, but that is not every possible bug. These tests can't see anything that only shows up with the real model: the chat template, the real tokenizer's EOS ids, LoRA, bf16 numerics, or running out of memory. The short runs against TRL are what catch those.

## Rules

- Don't edit a test to make it pass. If you think a test is wrong, write down why first.
- Read a failing test's message before its code. Most messages state the rule you broke.

"""The interface every advantage method implements, and the object the training loop calls.

A method is one file in src/advantages/, named after the method (vanilla.py, dr_grpo.py, ...), with:

    def advantages(batch: AdvantageBatch, state: dict, **params) -> Tensor  # [N] float32, one per completion

and, only if it needs them:

    NEEDS = ("seq_logps",)   # extra inputs the loop must compute (see AdvantageBatch)
    EXTRA_ROLLOUTS = 1       # completions sampled per prompt on top of the group size (MC-GRPO);
                             # they count against the rollout budget like any other
    GROUP_SIZE = 2           # the only group size the method is defined at (2-GRPO). The run script
                             # refuses any other budget.group_size

`params` are the method's own settings, from `advantage.params` in its config file. `state` is a
dict that persists from step to step (EBPO's global prior, KRPO's per-prompt filters, HA-DW's
anchor). Methods without state ignore it. Put only numbers, lists and tensors in it, so a run can
be saved and resumed.

A method may also return (advantages, keep): keep is a bool [N] marking the completions that enter
the loss. The others are masked out of it entirely (MC-GRPO drops the completion that is the median).

Nothing method-specific lives in src/train/. A method that also changes the loss (Dr. GRPO, DAPO)
does it through the `loss` section of its config file, not here.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import torch

KNOWN_NEEDS = {"seq_logps"}


@dataclass(frozen=True)
class AdvantageBatch:
    """Everything a method may look at. N = num_prompts * group_size completions.

    Rows i*G .. i*G+G-1 are the completions of prompt i (G = group_size, which includes any
    extra rollouts the method asked for).
    """
    rewards: torch.Tensor                     # [N] float32
    group_size: int                           # G
    prompt_ids: torch.Tensor                  # [N // G] long: each prompt's index in the training set,
                                              # the same every time that prompt comes back
    lengths: torch.Tensor                     # [N] counted completion tokens (up to and including EOS)
    step: int = 0                             # training step, from 0
    seq_logps: torch.Tensor | None = None     # [N] each completion's total log-prob under the model that
                                              # sampled it (LENS). Only filled if the method NEEDS it
    cluster_ids: torch.Tensor | None = None   # [N // G] each prompt's semantic cluster (BV-Blend)

    def __post_init__(self):
        n, g = self.rewards.numel(), self.group_size
        if self.rewards.dim() != 1 or g < 1 or n % g:
            raise ValueError(f"rewards must be 1-D with a length divisible by group_size, got {n} and {g}")
        if self.prompt_ids.numel() != n // g:
            raise ValueError(f"need one prompt id per group: {n // g}, got {self.prompt_ids.numel()}")

    @property
    def num_prompts(self) -> int:
        return self.rewards.numel() // self.group_size

    def grouped(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """x (default: rewards) as [num_prompts, G], one row per prompt."""
        return (self.rewards if x is None else x).view(self.num_prompts, self.group_size)


@dataclass
class Advantage:
    """A configured method: its function, its parameters, and its state across steps."""
    name: str
    fn: object
    params: dict = field(default_factory=dict)
    needs: tuple = ()
    extra_rollouts: int = 0
    group_size: int | None = None      # set only by methods defined at one group size (2-GRPO)
    state: dict = field(default_factory=dict)

    def check_group_size(self, group_size: int) -> None:
        if self.group_size is not None and group_size != self.group_size:
            raise ValueError(f"{self.name} is defined only at group size {self.group_size}; "
                             f"run it with --set budget.group_size={self.group_size}, not {group_size}")

    def __call__(self, batch: AdvantageBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(advantages [N] float32, keep [N] bool or None)."""
        out = self.fn(batch, self.state, **self.params)
        adv, keep = out if isinstance(out, tuple) else (out, None)
        adv = torch.as_tensor(adv, dtype=torch.float32)
        if adv.shape != batch.rewards.shape:
            raise ValueError(f"{self.name}: advantages must have shape {tuple(batch.rewards.shape)}, got {tuple(adv.shape)}")
        if not torch.isfinite(adv).all():
            raise FloatingPointError(f"{self.name}: non-finite advantage at step {batch.step}")
        if keep is not None:
            keep = torch.as_tensor(keep, dtype=torch.bool)
            if keep.shape != adv.shape:
                raise ValueError(f"{self.name}: keep must have shape {tuple(adv.shape)}")
        return adv, keep


def build_advantage(name: str, params: dict | None = None) -> Advantage:
    """Load src/advantages/<name>.py and bind its parameters."""
    if not name.isidentifier() or name in {"base", "grpo"}:
        raise ValueError(f"not an advantage method: {name!r}")
    try:
        module = importlib.import_module(f"src.advantages.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"no advantage method {name!r}: expected src/advantages/{name}.py") from e
    fn = getattr(module, "advantages", None)
    if not callable(fn):
        raise ValueError(f"src/advantages/{name}.py must define advantages(batch, state, **params)")
    needs = tuple(getattr(module, "NEEDS", ()))
    unknown = set(needs) - KNOWN_NEEDS
    if unknown:
        raise ValueError(f"{name}: the loop cannot provide {sorted(unknown)}; it knows {sorted(KNOWN_NEEDS)}")
    group_size = getattr(module, "GROUP_SIZE", None)
    return Advantage(name=name, fn=fn, params=dict(params or {}), needs=needs,
                     extra_rollouts=int(getattr(module, "EXTRA_ROLLOUTS", 0)),
                     group_size=None if group_size is None else int(group_size))

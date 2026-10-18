"""Vanilla GRPO's advantage. From week 11, every other method gets its own file in this folder."""
import torch


def group_advantages(rewards, group_size: int, scale: str = "group") -> torch.Tensor:
    """One advantage per completion: its reward minus the mean reward of its group.

    rewards: a list or 1-D tensor of length num_prompts * group_size. Entries i*G .. i*G+G-1
        are the G completions of prompt i (G = group_size).
    scale="group": also divide by (the group's standard deviation + 1e-4). Use the sample standard
        deviation, with Bessel's correction: that is what torch.std and TRL use.
    scale="none": don't divide. This is Dr. GRPO's choice.

    Returns a float32 tensor of the same length. A group whose rewards are all equal gets zeros.
    """
    r = torch.as_tensor(rewards, dtype=torch.float32).flatten()
    if group_size < 2 or r.numel() % group_size != 0:
        raise ValueError(f"need group_size >= 2 dividing {r.numel()} rewards, got group_size={group_size}")
    # One row per prompt: row i holds the G completions of prompt i (they are consecutive).
    r = r.view(-1, group_size)
    # The baseline is the group's own mean, so no critic network is needed (the point of GRPO).
    adv = r - r.mean(dim=1, keepdim=True)
    if scale == "group":
        # torch.std uses Bessel's correction (divides by G - 1), like TRL. The 1e-4 keeps a group
        # whose rewards are all equal at 0 / 1e-4 = 0 instead of 0 / 0 = NaN.
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-4)
    elif scale != "none":
        raise ValueError(f"unknown scale {scale!r}; use 'group' or 'none'")
    return adv.flatten()

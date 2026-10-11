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
    raise NotImplementedError

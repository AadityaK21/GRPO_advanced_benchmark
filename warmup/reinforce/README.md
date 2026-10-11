# REINFORCE test harness

For Project 1, weeks 1–2. You write `reinforce.py`, and these tests tell you when it's right. They check properties, not one particular implementation, so any correct REINFORCE passes.

## Setup

```
pip install torch gymnasium numpy pytest
```

## Run

```
pytest            # 18 fast checks, a few seconds
pytest -m slow    # trains on CartPole with three seeds, a few minutes on a laptop CPU
```

Right now all 18 fast checks fail with `NotImplementedError`. That's the starting point.

## Work in this order

| Step | You write | Tests that check it | What it teaches |
|---|---|---|---|
| 1 | `discounted_returns` | `test_returns_*` | the return G_t, computed in one backward pass |
| 2 | `Policy`, `select_action` | `test_policy_*`, `test_select_action_*` | sampling from a policy, and keeping log π(a) attached to the autograd graph |
| 3 | `reinforce_loss` | `test_loss_*`, `test_no_gradient_*`, `test_gradient_step_*` | the policy-gradient loss, its sign, and why the advantage is a constant |
| 4 | nothing new | `test_estimate_is_unbiased_*`, `test_baseline_reduces_variance` | why a baseline leaves the gradient unbiased but cuts its variance (your "After week 4" checkpoint) |
| 5 | `train` | `test_train_is_reproducible`, then `pytest -m slow` | the full loop, seeding, and what learning looks like |

## Rules

- Don't edit a test to make it pass. If you think a test is wrong, write down why first.
- Try to pass the fast tests before reading anyone else's REINFORCE code.
- The bandit tests use a fixed random generator, so a correct `reinforce_loss` gives the same numbers on every run.

## What the tests catch

These bugs were each injected into a working solution, and each one failed at least one test:

- greedy actions instead of sampling
- a detached log-prob
- no detach on the advantage
- sum instead of mean
- a flipped sign
- normalising inside the loss
- ignoring the baseline
- a missing discount
- a quadratic-time return loop
- an unseeded environment
- a missing batch dimension

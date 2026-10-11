# PPO test harness

For Project 1, weeks 3–4. Do the REINFORCE harness first. You write `ppo.py`, and these tests tell you when it's right. They check properties, not one particular implementation, so any correct PPO passes.

## Setup

```
pip install torch gymnasium numpy pytest
```

## Run

```
pytest            # 17 fast checks, a few seconds
pytest -m slow    # trains on CartPole with three seeds, a few minutes
```

Right now all 17 fast checks fail with `NotImplementedError`. That's the starting point.

## Work in this order

| Step | You write | Tests that check it | What it teaches |
|---|---|---|---|
| 1 | `compute_gae` | `test_gae_*` | advantages from a critic: TD errors, what λ does, and why nothing flows back across an episode end |
| 2 | `ActorCritic` | `test_actor_critic_shapes` | a policy head and a value head |
| 3 | `ppo_policy_loss` | `test_policy_loss_*`, `test_clipping_*`, `test_policy_gradient_*`, `test_no_gradient_*` | the clipped ratio: when it stops the gradient, and when it must not |
| 4 | `value_loss` | `test_value_loss_*` | training the critic |
| 5 | `train` | `test_train_is_reproducible`, then `pytest -m slow` | rollouts, GAE, and several epochs of minibatch updates |

Steps 1 and 3 matter most for later: GRPO keeps PPO's clipped ratio and swaps the critic for group statistics.

## Then compare with CleanRL

Once your CartPole run learns, compare your learning curves with CleanRL's `ppo.py` over 5 seeds, as the plan says. Read their code after your tests pass, not before.

## Rules

- Don't edit a test to make it pass. If you think a test is wrong, write down why first.
- If training feels slow on your laptop, call `torch.set_num_threads(1)` at the start of `train`. Small networks often run faster on one thread.

## What the tests catch

These bugs were each injected into a working solution, and each one failed at least one test:

- GAE without the episode-end mask in the TD error
- GAE without the episode-end mask in the recursion
- GAE ignoring the bootstrap value
- GAE missing gamma in the recursion
- returns without the values added back
- max instead of min in the clipped loss
- no clipping at all
- a flipped sign
- the ratio computed as a difference instead of exp of a difference
- no detach on the advantages
- the value loss without its 0.5
- values shaped [B, 1] instead of [B]
- an unseeded environment

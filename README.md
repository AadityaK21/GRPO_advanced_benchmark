# Project 1: GRPO advantage benchmark

The plan lives in the doc **Project 1 — GRPO Advantage Benchmark**:
https://claude.ai/artifact/1DoF6ogcFjsnxX5WeUuV9U

This folder becomes the repo.

## What's here

- `warmup/reinforce/`: weeks 1–2. Write `reinforce.py` until its tests pass. **Start here.**
- `warmup/ppo/`: weeks 3–4. PPO from scratch, the same way.
- `src/` and `tests/`: weeks 5–7. Your own GRPO loop, checked piece by piece and then against TRL. The guide is `docs/grpo-loop.md`.
- `docs/implementation-notes.md`: the logbook for every method's open questions, your choices, and author replies. Fill in the shared-loop settings while you build your loop (weeks 5–9).

Run `pytest` from this folder for the GRPO tests. Each warm-up has its own tests: run `pytest` inside its folder.

## What comes next

Weeks 8–9: the harness (configs, the swappable advantage interface, run logging), following the repo layout in the plan.

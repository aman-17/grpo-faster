# GRPO Async Training Pipeline

## Overview

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm used to fine-tune LLMs.

In this assignment, you're given a simulated GRPO training loop with a working but unoptimized baseline implementation. Your goal is to implement a more efficient async pipeline that improves throughput.

## Rollouts

Each rollout is a single unit of work that flows through three stages:

1. **Inference** (on GPU) — rollout goes from `CREATED` to `INFERENCED`
2. **Backward** (on GPU) — rollout goes from `INFERENCED` to `BACKWARDED`
3. **Optimizer step** — completed rollouts are collected into a batch and recorded

A training run has many rollouts (e.g. 12,800), processed in batches (e.g. 256 per optimizer step).

## GPUs

Each GPU has a fixed number of slots (e.g. 64) and can do **either** inference or backward at any given time:

- **Inference** — streams rollouts concurrently, each using 1 slot
- **Backward** — processes a batch of rollouts, locking **all** slots (exclusive access)

## The Baseline

`train_baseline` processes one batch at a time — all inference, then all backward, then optimizer step — before moving to the next:

All GPUs do the same phase at the same time. While inference runs, no backward work happens and vice versa.

## What to Implement

Open `grpo.py` and implement `train_async` to improve throughput while respecting all constraints. Read the TODO docstring above `train_async` and the support class docstrings carefully — they describe the constraints, key concepts, and hints.

## Running

```bash
python grpo.py
```

The harness runs both `train_baseline` and `train_async`, validates correctness, and reports the speedup.

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

# How to Run

No dependencies needed for local runs (Python 3.12+). Modal is only needed for cloud benchmarks.

## Run a single recipe

Each recipe is self-contained and compares itself against the baseline:

```bash
python recipe_01_colocated_hybrid.py
python recipe_02_agent_loop.py
python recipe_03_one_step_off.py
python recipe_04_fully_async.py     # best: 1.76x, also sweeps staleness thresholds
```

## Run all recipes locally

```bash
python bench_all.py                          # seed 0, time_scale 10000
python bench_all.py --seed 1 --repeat 3      # median of 3 repeats
python bench_all.py --time-scale 2000        # slower but higher fidelity
```

Trust the run only if the header says timer fidelity is below ~1.05x and no
recipe prints `CONTAMINATED` next to its CPU-busy line. If either appears,
lower `--time-scale` (each run gets proportionally slower but more accurate).

## Results

Verified at `--time-scale 2000` (timer inflation 1.01x, seed spread < 0.2%):

| Recipe              | Speedup vs baseline | Notes                                    |
| ------------------- | ------------------- | ---------------------------------------- |
| 01 colocated_hybrid | 1.17x               | fails validation on seed 2 (44/50 updates) |
| 02 agent_loop       | 1.01x               |                                          |
| 03 one_step_off     | 0.98x               |                                          |
| 04 fully_async      | **1.76x**           | best; ~0.15% off the provable ceiling    |

A run is valid only if all 12,800 rollouts are recorded and the model reaches
at least 45/50 weight updates; failures print the validation error instead of
a time.


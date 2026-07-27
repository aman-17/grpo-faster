"""Run the baseline and all four recipes against one fixed rollout set.

Usage:
    python bench_all.py                    # default: high-fidelity settings
    python bench_all.py --seed 7           # different latency draw
    python bench_all.py --repeat 3         # median of 3 draws
    python bench_all.py --time-scale 100000 --no-timer-fix    # reproduce the trap


READ THIS BEFORE TRUSTING ANY NUMBER FROM THIS SIMULATOR
========================================================
The simulator measures wall-clock and multiplies by `time_scale`. At the stock
`time_scale=100_000`, a 233ms simulated latency becomes a **2.33ms**
`asyncio.sleep`, and on Windows the default timer resolution is ~15.6ms. Measured
on this machine with a ProactorEventLoop:

    requested   0.60ms  ->  actual   6.19ms   (x10.3)
    requested   2.33ms  ->  actual   6.72ms   (x2.9)
    requested  23.30ms  ->  actual  26.22ms   (x1.13)

That is a roughly *additive* ~4ms floor per sleep. It does not scale with the
simulated latency, so it does not penalise runs uniformly -- it penalises them in
proportion to how many sleeps sit on the critical path. Which is precisely the
axis every pipelined design moves along: the baseline serialises 100 long sleeps
(50 inference waves + 50 backward phases), while a streaming pipeline serialises
~100 *short* ones per slot. Benchmarked naively, the stock config reports the
pipelined recipes as 0.7-0.9x, i.e. slower than the baseline, entirely as an
artifact.

Two corrections, both applied by default:

  1. `timeBeginPeriod(1)` (winmm) drops the system timer to 1ms.
  2. `time_scale=10_000` puts the median sleep at ~23ms, where the residual
     error is ~5% and, more importantly, is proportional rather than additive.

`run_and_report` normalises by `time_scale`, so the reported "Real time" stays
directly comparable across scales -- only the fidelity changes. Any conclusion
that flips between --time-scale 10000 and 100000 is a measurement artifact, not
a result. Cross-check with --repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import random
import statistics
import sys
import time

from grpo import (
    GPU,
    Config,
    GradientOptimizer,
    ModelState,
    make_rollouts,
    run_and_report,
    train_baseline,
)
from recipe_common import CapacityModel

import recipe_01_colocated_hybrid as r01
import recipe_02_agent_loop as r02
import recipe_03_one_step_off as r03
import recipe_04_fully_async as r04

RECIPES = [
    ("train_baseline", train_baseline, None),
    ("01 colocated_hybrid", r01.train_colocated_hybrid, r01),
    ("02 agent_loop", r02.train_agent_loop, r02),
    ("03 one_step_off", r03.train_one_step_off, r03),
    ("04 fully_async", r04.train_fully_async, r04),
]


@contextlib.contextmanager
def high_resolution_timer(enabled: bool):
    """Drop the Windows system timer to 1ms for the duration of the benchmark.

    No-op off Windows (Linux hrtimers are already fine). Always paired with
    timeEndPeriod -- leaving the global timer at 1ms costs battery machine-wide.
    """
    if not enabled or sys.platform != "win32":
        yield
        return
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)


async def measure_sleep_error(cfg: Config) -> float:
    """Report the residual timer inflation at the median simulated latency."""
    target = cfg.sim((cfg.inference_latency_lo + cfg.inference_latency_hi) / 3)
    import time as _time

    t0 = _time.perf_counter()
    for _ in range(50):
        await asyncio.sleep(target)
    return ((_time.perf_counter() - t0) / 50) / target


async def one_pass(cfg: Config) -> dict[str, float | None]:
    rollouts = make_rollouts(cfg)
    model_state = ModelState()
    gpus = [GPU(id=i, cfg=cfg, model_state=model_state) for i in range(cfg.num_gpus)]
    optimizer = GradientOptimizer(batch_size=cfg.batch_size, model_state=model_state)

    cap = CapacityModel.from_rollouts(rollouts, cfg)
    print(f"\n  capacity model: {cap.describe(cfg.num_gpus, cfg.num_rollouts)}")

    results: dict[str, float | None] = {}
    for name, fn, mod in RECIPES:
        cpu0 = time.process_time()
        real_s = await run_and_report(name, fn, rollouts, gpus, optimizer, cfg)
        cpu = time.process_time() - cpu0
        if mod is not None:
            line = mod.METRICS.line()
            if line:
                print(f"  Metrics    : {line}")
        if real_s is not None:
            frac = cpu / (real_s / cfg.time_scale)
            flag = "   <-- CONTAMINATED, lower --time-scale" if frac > 0.10 else ""
            print(f"  CPU busy   : {100 * frac:.1f}% of wall{flag}")
        results[name] = real_s
    return results


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--time-scale", type=int, default=10_000)
    ap.add_argument("--no-timer-fix", action="store_true")
    args = ap.parse_args()

    cfg = Config(time_scale=args.time_scale)

    with high_resolution_timer(not args.no_timer_fix):
        err = await measure_sleep_error(cfg)
        print(f"\n  timer fidelity: median sleep inflated x{err:.2f}"
              f"  (time_scale={cfg.time_scale}, timer_fix={not args.no_timer_fix})")
        if err > 1.25:
            print("  WARNING: inflation above 1.25x -- differences under ~20% are not meaningful.")

        all_runs: list[dict[str, float | None]] = []
        for i in range(args.repeat):
            random.seed(args.seed + i)
            all_runs.append(await one_pass(cfg))

    print("\n" + "=" * 68)
    print(f"  SUMMARY  ({args.repeat} draw(s), seed {args.seed}, time_scale={cfg.time_scale})")
    print("=" * 68)
    print(f"  {'recipe':<24}{'real time':>14}{'rollouts/min':>16}{'speedup':>12}")
    print("  " + "-" * 64)

    base_vals = [r["train_baseline"] for r in all_runs if r["train_baseline"] is not None]
    base = statistics.median(base_vals) if base_vals else None

    for name, _, _ in RECIPES:
        vals = [r[name] for r in all_runs if r[name] is not None]
        if not vals:
            print(f"  {name:<24}{'FAILED VALIDATION':>42}")
            continue
        med = statistics.median(vals)
        thru = cfg.num_rollouts / (med / 60)
        speedup = f"{base / med:.2f}x" if base else "-"
        print(f"  {name:<24}{med:>12.1f}s{thru:>16.0f}{speedup:>12}")
    print("=" * 68)


if __name__ == "__main__":
    asyncio.run(main())

"""Run the recipe benchmark on Modal, one dedicated CPU container per seed.

    pip install modal && modal setup
    modal run modal_bench.py                                  # seeds 0,1,2
    modal run modal_bench.py --seeds 0,1,2,3,4 --time-scale 2000
    modal run modal_bench.py --repeat 3                       # repeats per seed

WHY CPU-ONLY
------------
There is no GPU work in this assignment. `GPU.run_inference` and
`GPU.run_backward` are `asyncio.sleep` against a semaphore -- the whole thing is
a single-threaded Python event loop. Attaching an accelerator would cost money
and change nothing. `cpu=2.0` is deliberate: the benchmark is single-threaded, so
the second core exists purely to keep the event loop off a contended scheduler.

WHAT MOVING OFF A LAPTOP ACTUALLY BUYS
--------------------------------------
Two measurement defects, both of which make this simulator report *pipelined
designs as slower than the baseline* if you do not control for them.

1. Timer granularity. The harness's stock time_scale=100_000 turns a 233ms
   simulated latency into a 2.33ms sleep. Measured on Windows 11 with a
   ProactorEventLoop:

       requested   0.60ms  ->  actual   6.19ms   (x10.3)
       requested   2.33ms  ->  actual   6.72ms   (x2.9)
       requested  23.30ms  ->  actual  26.22ms   (x1.13)

   That is roughly *additive* per sleep, so it penalises a design in proportion
   to how many sleeps sit on its critical path -- exactly the axis pipelining
   moves along. Linux hrtimers make this a non-issue; the residual is a few
   percent and proportional rather than additive.

2. CPU contamination. The simulator measures wall-clock of a Python program, so
   scheduler CPU is billed as GPU time. On a laptop with background load this
   moved identical code by 27% run to run while the baseline (which is almost
   pure sleep) stayed fixed. A dedicated container fixes it; `bench_all.py`
   prints the CPU-busy fraction per run so you can confirm rather than assume.

Fan-out is one container per seed, so seeds cannot contaminate each other -- a
long run on one seed does not steal CPU from another. Check the reported spread:
if the same recipe varies more than a few percent across repeats, the numbers are
still measuring the machine, not the design. Drop --time-scale until it settles.
"""

from __future__ import annotations

import pathlib
import statistics

import modal

LOCAL_DIR = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        LOCAL_DIR,
        remote_path="/root/bench",
        ignore=["__pycache__", "*.pyc", ".git", ".venv", "__MACOSX"],
    )
)

app = modal.App("grpo-async-recipes", image=image)


@app.function(cpu=2.0, memory=2048, timeout=3600)
def bench_seed(seed: int, time_scale: int, repeat: int) -> dict:
    """One seed, on its own container. Returns raw per-recipe timings."""
    import asyncio
    import random
    import statistics
    import sys

    sys.path.insert(0, "/root/bench")

    import bench_all
    from grpo import Config

    cfg = Config(time_scale=time_scale)

    async def go() -> dict:
        err = await bench_all.measure_sleep_error(cfg)
        runs = []
        for i in range(repeat):
            random.seed(seed)  # same rollout latencies every repeat
            runs.append(await bench_all.one_pass(cfg))
        merged = {}
        for name in runs[0]:
            vals = [r[name] for r in runs if r[name] is not None]
            merged[name] = statistics.median(vals) if vals else None
        return {"seed": seed, "sleep_error": err, "median": merged, "runs": runs}

    return asyncio.run(go())


@app.local_entrypoint()
def main(seeds: str = "0,1,2", time_scale: int = 10_000, repeat: int = 1) -> None:
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    args = [(s, time_scale, repeat) for s in seed_list]

    results = list(bench_seed.starmap(args))
    results.sort(key=lambda r: r["seed"])

    names = list(results[0]["median"].keys())
    print("\n" + "=" * 84)
    print(f"  MODAL RESULTS   seeds={seed_list}  time_scale={time_scale}  repeat={repeat}")
    print("=" * 84)
    print("  " + "recipe".ljust(24) + "".join(f"seed {r['seed']:<10}" for r in results) + "median speedup")
    print("  " + "-" * 80)

    baseline = [r["median"].get("train_baseline") for r in results]
    for name in names:
        cells, ratios = [], []
        for r, base in zip(results, baseline):
            v = r["median"].get(name)
            if v is None:
                cells.append("FAIL".ljust(15))
            else:
                cells.append(f"{v:>10.0f}     ")
                if base:
                    ratios.append(base / v)
        med = f"{statistics.median(ratios):.2f}x" if ratios else "-"
        print(f"  {name:<24}" + "".join(cells) + f"{med:>10}")

    errs = [r["sleep_error"] for r in results]
    print("  " + "-" * 80)
    print(f"  timer inflation: {min(errs):.2f}x - {max(errs):.2f}x   (want < 1.05x)")
    print("=" * 84)

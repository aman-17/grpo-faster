"""Recipe 03 -- One Step Off Policy  (disaggregated, fixed 1-step staleness)
===========================================================================

verl reference
--------------
    recipe/one_step_off_policy/
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes / rollout.n_gpus_per_node   vs   trainer.nnodes / trainer.n_gpus_per_node
    verl docs: advance/one_step_off
    Reported: +40% end-to-end on Qwen2.5-Math-7B / DAPO (18h21m -> 13h06m, Megatron)

What verl actually does
-----------------------
Stop colocating. Give generation its own GPUs and training the rest, then run
them concurrently one step out of phase:

    step N:   train on the sequences generated during step N-1
              *while* generating the sequences for step N+1

Because the two groups no longer share HBM, there is no reshard and no offload --
but the weights now live in two places, so after every optimizer step the actor
must push its parameters to the rollout group. verl does this over NCCL:
`get_actor_weights_info()` / `set_actor_weights_info()` exchange shapes and
dtypes, both groups join a collective built on a shared master addr/port, and
parameters are broadcast tensor by tensor -- typically under 300ms.

The cost is exactly one step of staleness, permanently. The tuning rule from the
docs is a single sentence: size the two groups so the two phases take comparable
wall-clock. The `wait_prev_gen` metric measures how badly you got that wrong.

How that maps onto this simulator
---------------------------------
Split the 4 GPUs into a generation group and a training group. `plan_split` in
recipe_common sizes them from the measured capacity of each role:

    inference : slots / mean_latency        ~275 rollouts/s/GPU
    backward  : slots / median_latency      ~298 rollouts/s/GPU

Near parity, so the split lands at 2/2. Note that the tempting 3-gen/1-train is
*worse than the baseline*: one backward GPU tops out around 298 rollouts/s while
three generation GPUs supply ~825/s, and the trainer becomes the wall.

The freshness trap
------------------
`ModelState.try_increment` only advances the version when the batch's *max*
rollout version is at least the current version, so one-step-off is not
automatically safe:

    batch b+1 is generated while batch b trains. If generation of b+1 finishes
    *before* optimizer_step(b) lands, every rollout in b+1 captured version b,
    the check `b >= b+1` fails, and the version pins forever.

The batch survives only because its stragglers are still decoding when the step
fires and pick up the new version. In other words: this recipe is correct only
while generation is the slower phase. Give training too few GPUs and throughput
looks fine but Validator Check 3 fails. `METRICS.stale_batches` counts it.

That fragility is inherent to fixing the batch membership before the batch is
generated. Recipe 04 removes it by not fixing membership at all.

Deliberately not used
---------------------
Same two simulator exploits declined in recipe 01: latency-oracle scheduling, and
gaming `backward_latency_for`'s median by composing chunks from extremes.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from grpo import GPU, GradientOptimizer, Rollout
from recipe_common import CapacityModel, Gate, Metrics, Signal, chunked

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


@dataclass
class OneStepOffConfig:
    gen_gpus: int | None = None
    """Size of the generation group. None -> plan_split from the capacity model."""

    spill_batches: int = 0
    """How far past the batch currently being generated the rollouters may run
    when they would otherwise idle on a straggler. 0 = strict one-step-off.

    Measured, not assumed: spill_batches=1 recovers the tail but breaks Check 3
    outright (25 of 50 batches fail to increment at this config). Running ahead
    means batch b+1 is fully generated *before* optimizer_step(b) fires, so every
    member captured version b and `b >= b+1` is false. The straggler you were
    trying to hide is the only thing that was keeping the batch fresh."""

    freshness_guard: int = 0
    """Rollouts per batch held back until after the *previous* optimizer step.

    An attempt to have both spill and freshness: if the last few rollouts of
    batch b+1 only start after step(b), they necessarily capture version b+1 and
    the batch increments. Implemented and measured -- see __main__. It does not
    rescue the design, because the trainer then blocks on those deliberately-late
    rollouts and the window grows by roughly a full inference latency."""

    param_sync_ms: float = 0.0
    """Simulated NCCL weight broadcast after each optimizer step, during which
    generation is stopped. verl measures this under 300ms in practice; the
    assignment's cost model has no such term, so it defaults to 0."""


# ---------------------------------------------------------------------------
# Rollouter  (the generation group)
# ---------------------------------------------------------------------------


class Rollouter:
    """Continuous generation over the gen group, gated by a batch horizon.

    One coroutine per slot, all pulling from a single ordered stream. That is the
    important structural detail: rollouts are *not* pre-assigned to GPUs. A slot
    that frees early immediately takes the next rollout, so a GPU that draws a
    slow tail does not sit idle while its neighbour has queued work.

    `horizon` is the exclusive rollout index the group is allowed to start, and
    the trainer advances it once per step. It is the only thing bounding how far
    generation runs ahead of training.
    """

    def __init__(
        self,
        gpus: list[GPU],
        rollouts: list[Rollout],
        batch_size: int,
        slots: int,
        metrics: Metrics,
    ) -> None:
        self._gpus = gpus
        self._rollouts = rollouts
        self._bs = batch_size
        self._slots = slots
        self._metrics = metrics
        self._issue: deque[int] = deque()
        self._guarded: dict[int, list[int]] = {}
        self._opened: set[int] = set()
        self._issued = 0
        self.paused = False

        # Two separate wakeup channels on purpose. `_work` is slot-scale (up to
        # num_gen_gpus * slots waiters) and must be woken by count; `progress` has
        # exactly one waiter (the trainer). Sharing one broadcast between them
        # wakes 128 parked slot workers on every single inference completion.
        self._work = Gate()
        self.progress = Signal()

        n_batches = (len(rollouts) + batch_size - 1) // batch_size
        self._remaining = [0] * n_batches
        for i in range(len(rollouts)):
            self._remaining[i // batch_size] += 1
        self._tasks: list[asyncio.Task] = []
        self._closed = False

    # -- admission control --------------------------------------------------

    def open_batch(self, b: int, guard: int) -> None:
        """Make batch `b` issuable, holding back its last `guard` rollouts.

        The held-back tail is what guarantees batch `b` contains a rollout that
        finished inference *after* optimizer_step(b-1) -- see `release_guard`.
        Idempotent: spill and the normal advance can both name the same batch.
        """
        if b in self._opened or b >= len(self._remaining):
            return
        self._opened.add(b)
        lo = b * self._bs
        hi = min(lo + self._bs, len(self._rollouts))
        keep = max(lo, hi - guard)
        self._enqueue(range(lo, keep))
        if keep < hi:
            self._guarded[b] = list(range(keep, hi))

    def release_guard(self, b: int) -> None:
        self._enqueue(self._guarded.pop(b, []))

    def _enqueue(self, indices) -> None:
        n = 0
        for i in indices:
            self._issue.append(i)
            n += 1
        if n:
            self._work.wake(n)

    # -- control ------------------------------------------------------------

    def start(self) -> None:
        for gpu in self._gpus:
            for _ in range(self._slots):
                self._tasks.append(asyncio.create_task(self._slot_worker(gpu)))

    def close(self) -> None:
        self._closed = True
        self._work.wake_all()

    async def pause_for_sync(self, seconds: float) -> None:
        """Weight broadcast: generation must stop while parameters are in flight."""
        self._metrics.param_syncs += 1
        if seconds <= 0:
            return
        self.paused = True
        await asyncio.sleep(seconds)
        self.paused = False
        self._work.wake_all()

    async def wait_batch(self, b: int) -> None:
        while self._remaining[b] > 0:
            await self.progress.wait()

    async def join(self) -> None:
        await asyncio.gather(*self._tasks)

    # -- worker -------------------------------------------------------------

    async def _slot_worker(self, gpu: GPU) -> None:
        while True:
            while self.paused or not self._issue:
                if self._closed and not self._issue:
                    return
                await self._work.wait()

            idx = self._issue.popleft()  # no await between pop and use
            await gpu.run_inference(self._rollouts[idx])

            self._issued += 1
            self._remaining[idx // self._bs] -= 1
            if self._remaining[idx // self._bs] == 0:
                self.progress.notify()


# ---------------------------------------------------------------------------
# Trainer  (the training group)
# ---------------------------------------------------------------------------


async def _backward_batch(
    batch: list[Rollout],
    train_gpus: list[GPU],
    slots: int,
    metrics: Metrics,
) -> None:
    """Backward every chunk of a batch across the training group.

    Work-stealing rather than the baseline's static `i % num_gpus` assignment:
    chunk latencies differ (each is priced at its own median), so a static
    assignment leaves GPUs idle at the tail of a step.

    Exactly one worker per GPU. `GPU.run_backward` acquires its permits one at a
    time, so two concurrent backwards on the same GPU can deadlock holding
    partial sets. This is a correctness constraint, not a tuning choice.
    """
    queue = deque(chunked(batch, slots))

    async def worker(gpu: GPU) -> None:
        while queue:
            chunk = queue.popleft()
            await gpu.run_backward(chunk)
            metrics.backward_calls += 1

    await asyncio.gather(*[worker(g) for g in train_gpus])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

METRICS = Metrics(name="train_one_step_off")
CONFIG = OneStepOffConfig()


async def train_one_step_off(
    rollouts: list[Rollout],
    gpus: list[GPU],
    batch_size: int,
    slots_per_gpu: int,
    optimizer: GradientOptimizer,
) -> None:
    """Disaggregated generation/training, overlapped one step out of phase."""
    global METRICS
    METRICS = Metrics(name="train_one_step_off")

    cap = CapacityModel.from_rollouts(rollouts, gpus[0].cfg)
    if CONFIG.gen_gpus is not None:
        n_gen = CONFIG.gen_gpus
    else:
        n_gen, _ = cap.plan_split(len(gpus), len(rollouts))
    gen_gpus, train_gpus = gpus[:n_gen], gpus[n_gen:]
    assert gen_gpus and train_gpus, "both groups need at least one GPU"
    METRICS.extra["split"] = f"{len(gen_gpus)}gen/{len(train_gpus)}train"
    METRICS.extra["spill"] = CONFIG.spill_batches

    METRICS.extra["guard"] = CONFIG.freshness_guard

    batches = chunked(rollouts, batch_size)
    rollouter = Rollouter(gen_gpus, rollouts, batch_size, slots_per_gpu, METRICS)
    rollouter.open_batch(0, guard=0)  # nothing to be fresh relative to yet
    rollouter.start()

    for b, batch in enumerate(batches):
        t0 = time.perf_counter()
        await rollouter.wait_batch(b)
        METRICS.wait_prev_gen_s += time.perf_counter() - t0

        # Open generation for the next step *before* training this one -- this
        # single reordering is the whole recipe.
        for k in range(b + 1, b + 2 + CONFIG.spill_batches):
            rollouter.open_batch(k, guard=CONFIG.freshness_guard)

        await _backward_batch(batch, train_gpus, slots_per_gpu, METRICS)

        version_before = await optimizer.model_state.get_weights_version()
        await optimizer.optimizer_step(batch)
        if await optimizer.model_state.get_weights_version() == version_before:
            METRICS.stale_batches += 1

        # Only now can batch b+1's held-back tail capture the new version.
        rollouter.release_guard(b + 1)

        await rollouter.pause_for_sync(gpus[0].cfg.sim(CONFIG.param_sync_ms))

    rollouter.close()
    await rollouter.join()


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from grpo import Config, ModelState, make_rollouts, run_and_report, train_baseline

    async def main() -> None:
        cfg = Config()
        rollouts = make_rollouts(cfg)
        model_state = ModelState()
        gpus = [GPU(id=i, cfg=cfg, model_state=model_state) for i in range(cfg.num_gpus)]
        optimizer = GradientOptimizer(batch_size=cfg.batch_size, model_state=model_state)

        cap = CapacityModel.from_rollouts(rollouts, cfg)
        print(f"\n  capacity model: {cap.describe(cfg.num_gpus, cfg.num_rollouts)}\n")

        base = await run_and_report("train_baseline", train_baseline, rollouts, gpus, optimizer, cfg)

        # Sweep every legal split -- the point is that the wrong one loses to the
        # baseline outright, which is exactly verl's "balance the two groups" rule.
        for n_gen in (1, 2, 3):
            CONFIG.gen_gpus = n_gen
            got = await run_and_report(
                f"one_step_off ({n_gen}gen/{cfg.num_gpus - n_gen}train)",
                train_one_step_off,
                rollouts,
                gpus,
                optimizer,
                cfg,
            )
            print(f"  Metrics    : {METRICS.line()}")
            if base and got:
                print(f"  Speedup    : {base / got:.2f}x")

    asyncio.run(main())

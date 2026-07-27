"""Recipe 04 -- Fully Async Policy Trainer  (streaming, tunable staleness)
=========================================================================

verl reference
--------------
    recipe/fully_async_policy/
    verl docs: advance/fully_async
    async_training.staleness_threshold
    async_training.require_batches
    async_training.trigger_parameter_sync_step
    async_training.partial_rollout
    Reported: 2.35x - 2.67x on 128-GPU runs

What verl actually does
-----------------------
Four components, fully decoupled:

    Rollouter              generates sample by sample onto a queue; its production
                           rate is throttled only by a freshness constraint
    MessageQueue           holds generated samples for the trainer
    Trainer                pulls `require_batches * ppo_mini_batch_size` samples,
                           trains, and triggers a parameter sync every
                           `trigger_parameter_sync_step` rounds
    ParameterSynchronizer  NCCL broadcast from trainer group to rollouter group

The difference from one-step-off is not "more overlap". It is that batch
membership is no longer decided before the batch exists. One-step-off says
"batch N+1 is these 256 prompts, tell me when all 256 are done" -- so every step
pays the full straggler tail. Fully-async says "give me the next 256 that
finish". Nobody waits for a straggler; the straggler simply lands in a later
batch.

Staleness stops being a constant and becomes a dial:

    max in-flight samples = (1 + staleness_threshold)
                            * trigger_parameter_sync_step * require_batches * mini_batch

with threshold 0 degenerating to synchronous training, and the docs recommending
you keep it below 1. `partial_rollout=True` additionally lets the Rollouter
interrupt a decode at a sync boundary, checkpoint the partial sequence, and
resume it after.

How that maps onto this simulator
---------------------------------
Rollouter = slot-level workers over the generation group, pushing INFERENCED
rollouts onto a bounded queue. Trainer = one worker per training GPU pulling
`slots_per_gpu` at a time, running backward, and handing the chunk to batch
assembly. The queue bound *is* the staleness threshold.

Why the queue bound is load-bearing (the derivation)
----------------------------------------------------
`ModelState.try_increment` advances the version only if the batch's max rollout
version is at least the current version. Batch b+1 therefore needs at least one
member that finished inference after optimizer_step(b) fired.

Because batches are assembled in backward-completion order and submitted the
instant they fill, the freshest member of a batch is the one that just finished
backward. Its age since inference is:

    lag  =  queue_depth / throughput  +  backward_latency

and the interval between optimizer steps is:

    period  =  batch_size / throughput

Freshness requires lag < period, i.e.

    queue_depth  <  batch_size - throughput * backward_latency

At this config (throughput ~550 rollouts/s, backward ~215ms, batch 256) that is
roughly 138 rollouts -- about half a batch, so staleness_threshold ~= 0.5. Which
is how verl's "keep it below 1" advice falls out of the arithmetic rather than
out of folklore. Raise `staleness_threshold` past that and throughput barely
moves while Check 3 fails; the sweep in __main__ shows the cliff.

No analogue available
---------------------
`partial_rollout` cannot be modelled here: `run_inference` is a single atomic
sleep, so a decode cannot be interrupted and resumed. The nearest legal thing --
declining to start a rollout just before a sync -- is a no-op while
`param_sync_ms` is 0, so it is not implemented rather than faked.

Deliberately not used
---------------------
Same two simulator exploits declined in recipes 01 and 03: latency-oracle
scheduling, and gaming `backward_latency_for`'s median by composing chunks from
extremes. Chunks here are strict FIFO off the queue, which is also what keeps
staleness bounded.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from grpo import GPU, GradientOptimizer, Rollout
from recipe_common import BatchRecorder, CapacityModel, Gate, Metrics, Signal

@dataclass
class FullyAsyncConfig:
    gen_gpus: int | None = None
    """Rollouter group size. None -> plan_split from the capacity model."""

    staleness_threshold: float = 0.5
    """Queue capacity as a fraction of batch_size. 0 -> synchronous. See the
    derivation above for why ~0.5 is the ceiling at this config."""

    trigger_parameter_sync_step: int = 1
    """Optimizer steps between parameter broadcasts."""

    param_sync_ms: float = 0.0
    """Simulated NCCL broadcast cost, during which the Rollouter is stopped."""


class MessageQueue:
    """Bounded FIFO of INFERENCED rollouts, with backpressure on the producer.

    Not `asyncio.Queue`, because the consumer needs to take exactly
    `slots_per_gpu` at once (a backward call is all-or-nothing) and needs to
    settle for a partial chunk once the producer has closed. Draining an
    asyncio.Queue item-by-item to build a chunk reintroduces a head-of-line stall.
    """

    def __init__(self, capacity: int, metrics: Metrics) -> None:
        self.capacity = capacity
        self._dq: deque[Rollout] = deque()
        self._not_full = Gate()  # producers blocked on the staleness bound
        self._not_empty = Gate()  # consumers blocked waiting for a full chunk
        self._closed = False
        self._metrics = metrics

    def __len__(self) -> int:
        return len(self._dq)

    def close(self) -> None:
        self._closed = True
        self._not_empty.wake_all()

    async def put(self, rollout: Rollout) -> None:
        while len(self._dq) >= self.capacity:
            await self._not_full.wait()  # backpressure == the staleness bound
        self._dq.append(rollout)
        self._metrics.peak_queue_depth = max(self._metrics.peak_queue_depth, len(self._dq))
        self._not_empty.wake(1)

    def _take(self, n: int) -> list[Rollout]:
        chunk = [self._dq.popleft() for _ in range(n)]
        self._not_full.wake(n)  # counted: exactly n producers can now proceed
        return chunk

    async def get_chunk(self, n: int) -> list[Rollout] | None:
        while True:
            if len(self._dq) >= n:
                return self._take(n)
            if self._closed:
                return self._take(len(self._dq)) if self._dq else None
            await self._not_empty.wait()


class ParameterSynchronizer:
    """NCCL broadcast trainer -> rollouter, every `trigger_parameter_sync_step`."""

    def __init__(self, rollouter: "Rollouter", cfg: FullyAsyncConfig, sim, metrics: Metrics) -> None:
        self._rollouter = rollouter
        self._cfg = cfg
        self._sim = sim
        self._metrics = metrics

    async def on_optimizer_step(self, step: int) -> None:
        if step % self._cfg.trigger_parameter_sync_step:
            return
        self._metrics.param_syncs += 1
        if self._cfg.param_sync_ms <= 0:
            return
        self._rollouter.paused = True
        await asyncio.sleep(self._sim(self._cfg.param_sync_ms))
        self._rollouter.paused = False
        self._rollouter.changed.notify()


class Rollouter:
    """Sample-by-sample generation over the gen group, into the MessageQueue.

    One coroutine per slot pulling from one shared stream: a slot that frees
    early takes the next rollout immediately. There is no batch boundary here at
    all, which is precisely what removes the per-step straggler barrier.
    """

    def __init__(self, gpus: list[GPU], rollouts: list[Rollout], slots: int, queue: MessageQueue) -> None:
        self._gpus = gpus
        self._rollouts = rollouts
        self._slots = slots
        self._queue = queue
        self._next = 0
        self._live = 0
        self.paused = False
        self.changed = Signal()

    async def run(self) -> None:
        workers = [
            asyncio.create_task(self._slot_worker(gpu)) for gpu in self._gpus for _ in range(self._slots)
        ]
        await asyncio.gather(*workers)
        self._queue.close()

    async def _slot_worker(self, gpu: GPU) -> None:
        while True:
            while self.paused:
                await self.changed.wait()
            if self._next >= len(self._rollouts):
                return
            rollout = self._rollouts[self._next]
            self._next += 1  # no await between read and bump -> no double-issue
            await gpu.run_inference(rollout)
            await self._queue.put(rollout)


async def _trainer_worker(
    gpu: GPU,
    queue: MessageQueue,
    recorder: BatchRecorder,
    slots: int,
    metrics: Metrics,
) -> None:
    """One per training GPU -- and only one, or concurrent `run_backward` calls
    on the same GPU deadlock holding partial permit sets."""
    while True:
        chunk = await queue.get_chunk(slots)
        if chunk is None:
            return
        await gpu.run_backward(chunk)
        metrics.backward_calls += 1
        await recorder.record(chunk)

METRICS = Metrics(name="train_fully_async")
CONFIG = FullyAsyncConfig()


async def train_fully_async(
    rollouts: list[Rollout],
    gpus: list[GPU],
    batch_size: int,
    slots_per_gpu: int,
    optimizer: GradientOptimizer,
) -> None:
    """Rollouter -> bounded MessageQueue -> Trainer, with staleness as a dial."""
    global METRICS
    METRICS = Metrics(name="train_fully_async")

    cfg = gpus[0].cfg
    cap = CapacityModel.from_rollouts(rollouts, cfg)
    n_gen = CONFIG.gen_gpus if CONFIG.gen_gpus is not None else cap.plan_split(len(gpus), len(rollouts))[0]
    gen_gpus, train_gpus = gpus[:n_gen], gpus[n_gen:]
    assert gen_gpus and train_gpus, "both groups need at least one GPU"

    capacity = max(slots_per_gpu, int(CONFIG.staleness_threshold * batch_size))
    METRICS.extra["split"] = f"{len(gen_gpus)}gen/{len(train_gpus)}train"
    METRICS.extra["queue_cap"] = capacity

    queue = MessageQueue(capacity, METRICS)
    rollouter = Rollouter(gen_gpus, rollouts, slots_per_gpu, queue)
    syncer = ParameterSynchronizer(rollouter, CONFIG, cfg.sim, METRICS)
    recorder = BatchRecorder(optimizer, batch_size, on_step=syncer.on_optimizer_step)

    await asyncio.gather(
        rollouter.run(),
        *[_trainer_worker(g, queue, recorder, slots_per_gpu, METRICS) for g in train_gpus],
    )
    await recorder.close()
    METRICS.stale_batches = recorder.stale_steps


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

        print("\n  --- staleness sweep (throughput flattens; correctness falls off a cliff) ---")
        for threshold in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            CONFIG.staleness_threshold = threshold
            got = await run_and_report(
                f"fully_async (staleness_threshold={threshold})",
                train_fully_async,
                rollouts,
                gpus,
                optimizer,
                cfg,
            )
            print(f"  Metrics    : {METRICS.line()}")
            if base and got:
                print(f"  Speedup    : {base / got:.2f}x")

    asyncio.run(main())

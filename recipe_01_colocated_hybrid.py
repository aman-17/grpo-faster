"""Recipe 01 -- Colocated HybridEngine  (verl's default placement)
=================================================================

verl reference
--------------
    actor_rollout_ref.hybrid_engine=True            (the default)
    verl/workers/sharding_manager/{fsdp_vllm,megatron_vllm}.py
    verl/trainer/ppo/ray_trainer.py :: RayPPOTrainer.fit()
    HybridFlow paper, S4 "3D-HybridEngine"          arxiv.org/abs/2409.19256

What verl actually does
-----------------------
Every GPU hosts *both* engines: the training engine (FSDP / Megatron) and the
inference engine (vLLM / SGLang). There is no static partition of the cluster --
the whole cluster generates, then the whole cluster trains. verl's contribution
here is not removing the barrier, it is making the barrier *cheap to cross*:

    train -> generate :  offload optimizer state, all-gather the training shards,
                         re-slice into the inference parallel layout, vLLM wake_up()
                         so the KV cache can claim the freed HBM
    generate -> train :  vLLM sleep(), drop KV cache, reload optimizer state

3D-HybridEngine does that reshard with zero memory redundancy. Because the
transition still costs real seconds, the design pressure is: cross it as *rarely*
as your staleness budget allows.

How that maps onto this simulator
---------------------------------
The simulator charges nothing for resharding -- but it charges the *other* half
of the transition cost, and that half turns out to be the expensive one.

`GPU.run_backward` acquires all `slots_per_gpu` semaphore permits before it can
start. So a GPU cannot switch into backward until every in-flight inference on it
has retired. Occupancy decays from 64 to 0 while the longest straggler finishes,
and with latencies drawn from Beta(1.5, 3) on [50, 600] ms, that straggler is
frequently ~600 ms while the mean is ~233 ms. *That drain is this simulator's
sleep()/wake_up().*

Design
------
Each GPU runs one scheduler coroutine -- an explicit three-state machine, which
is roughly the shape of a real inference-engine step loop:

    INFER    admit rollouts until all slots are busy; retire completions into
             the shared ready-pool
    DRAIN    stop admitting; keep retiring until in-flight hits zero
             (this is the transition cost -- occupancy is decaying, nothing
             refills it)
    BACKWARD hold all slots, run `chunks_per_flush` backward calls back to back,
             then return to INFER

`chunks_per_flush` is the amortisation knob and it is the entire tension of this
recipe, exactly as in verl:

    larger  -> fewer drains, less wasted occupancy
    larger  -> deeper ready-pool -> rollouts sit longer between inference and
               being recorded -> their captured weights_version goes stale ->
               Validator Check 3 fails

`max_ready` is the staleness bound (verl: `staleness_threshold`). Inference
admission blocks when the ready-pool is full; that backpressure is the only thing
keeping weights_version healthy.

Deliberately not used
---------------------
* `Rollout.inference_latency` is visible before the rollout runs. Scheduling on it
  (shortest-job-first, drain-aware admission) would be a large win here and is
  pure oracle -- in production you do not know a sequence's output length before
  you decode it. Not used.
* `backward_latency_for` costs a chunk at its *median* latency, so a chunk of
  "33 tiny + 31 huge" prices as tiny. Chunk composition can therefore be gamed to
  drive simulated backward cost toward the floor. That is an artifact of the cost
  model, not a real optimisation (real backward cost tracks total tokens, not the
  median). Chunks here are taken in FIFO completion order. Not used.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from grpo import GPU, GradientOptimizer, Rollout
from recipe_common import BatchRecorder, Metrics, Signal

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


@dataclass
class HybridEngineConfig:
    chunks_per_flush: int = 2
    """Backward calls per role switch. Amortises the drain; costs queue depth."""

    max_ready: int | None = None
    """Staleness bound, in rollouts. Defaults to batch_size // 2 -- see the
    derivation in recipe 04: a rollout must reach optimizer_step within one batch
    period of finishing inference, which at this config caps the queue near half a
    batch. Admission blocks above this. verl: `async_training.staleness_threshold`.

    Note the bound is a *watermark*, not a hard cap: up to num_gpus*slots
    inferences are already in flight when admission closes and they still land in
    the pool. Expect peak_q to overshoot by roughly that much."""

    switch_cost_ms: float = 0.0
    """Simulated 3D-HybridEngine reshard + offload cost, charged on each role
    switch. Zero by default because the assignment's cost model has no such term;
    raise it to see why verl eventually disaggregated (see __main__)."""


# ---------------------------------------------------------------------------
# Shared cluster state
# ---------------------------------------------------------------------------


class _Cluster:
    """Everything the per-GPU schedulers coordinate through.

    Single-threaded asyncio, so no locks are needed for the plain counters --
    only for the awaits. Mutations are grouped so no `await` ever splits an
    invariant.
    """

    def __init__(
        self,
        rollouts: list[Rollout],
        slots: int,
        recorder: BatchRecorder,
        cfg: HybridEngineConfig,
        max_ready: int,
        metrics: Metrics,
    ) -> None:
        self.pending: deque[Rollout] = deque(rollouts)
        self.ready: deque[Rollout] = deque()
        self.slots = slots
        self.total = len(rollouts)
        self.backwarded = 0
        self.recorder = recorder
        self.cfg = cfg
        # A flush can never require more queued rollouts than the queue is
        # allowed to hold, or admission backpressure and the flush trigger
        # deadlock against each other.
        self.max_ready = max(max_ready, slots)
        self.flush_watermark = min(cfg.chunks_per_flush * slots, self.max_ready)
        self.metrics = metrics
        self.changed = Signal()

    # -- predicates ---------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.backwarded >= self.total

    @property
    def tail(self) -> bool:
        """No inference work left to admit anywhere -- flush partial chunks now."""
        return not self.pending

    def can_admit(self) -> bool:
        return bool(self.pending) and len(self.ready) < self.max_ready

    def should_flush(self) -> bool:
        if len(self.ready) >= self.flush_watermark:
            return True
        # Tail: production has stopped, so waiting for a full flush deadlocks.
        return self.tail and bool(self.ready)

    # -- mutations ----------------------------------------------------------

    def take(self) -> Rollout | None:
        if not self.can_admit():
            return None
        return self.pending.popleft()

    def retire(self, rollout: Rollout) -> None:
        self.ready.append(rollout)
        self.metrics.peak_queue_depth = max(self.metrics.peak_queue_depth, len(self.ready))
        self.changed.notify()

    def claim(self) -> list[list[Rollout]]:
        """Pop up to `chunks_per_flush` backward chunks off the ready-pool.

        Popping *before* the drain starts is what prevents two GPUs from claiming
        the same rollouts, and means a GPU never drains for work it won't get.
        """
        chunks: list[list[Rollout]] = []
        for _ in range(self.cfg.chunks_per_flush):
            if not self.ready:
                break
            n = min(self.slots, len(self.ready))
            if n < self.slots and not self.tail:
                break  # partial chunks only at the tail; otherwise wait for a full one
            chunks.append([self.ready.popleft() for _ in range(n)])
        if chunks:
            self.changed.notify()  # freed queue space -> unblock admission
        return chunks

    def on_backwarded(self, n: int) -> None:
        self.backwarded += n
        self.changed.notify()


# ---------------------------------------------------------------------------
# Per-GPU scheduler
# ---------------------------------------------------------------------------


async def _infer_one(gpu: GPU, rollout: Rollout, cl: _Cluster) -> None:
    await gpu.run_inference(rollout)
    cl.retire(rollout)


async def _gpu_scheduler(gpu: GPU, cl: _Cluster) -> None:
    inflight: set[asyncio.Task] = set()
    draining = False

    while True:
        if cl.done and not inflight:
            return

        # ---- role decision ------------------------------------------------
        if not draining and cl.should_flush():
            draining = True
            cl.metrics.role_switches += 1
            drain_t0 = time.perf_counter()

        if draining and not inflight:
            cl.metrics.drain_s += time.perf_counter() - drain_t0
            chunks = cl.claim()
            draining = False
            if chunks:
                if cl.cfg.switch_cost_ms:
                    await asyncio.sleep(gpu.cfg.sim(cl.cfg.switch_cost_ms))
                for chunk in chunks:
                    await gpu.run_backward(chunk)
                    cl.metrics.backward_calls += 1
                    cl.on_backwarded(len(chunk))
                    await cl.recorder.record(chunk)
                if cl.cfg.switch_cost_ms:
                    await asyncio.sleep(gpu.cfg.sim(cl.cfg.switch_cost_ms))
                continue
            # Nothing claimable after all (another GPU beat us to it) -> fall
            # through and go back to admitting inference.

        # ---- admission ----------------------------------------------------
        if not draining:
            while len(inflight) < gpu.slots_per_gpu:
                rollout = cl.take()
                if rollout is None:
                    break
                inflight.add(asyncio.create_task(_infer_one(gpu, rollout, cl)))

        # ---- block until something changes --------------------------------
        if inflight:
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()  # re-raise; a swallowed assert here would surface as a hang
        else:
            if cl.done:
                return
            await cl.changed.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

METRICS = Metrics(name="train_colocated_hybrid")
CONFIG = HybridEngineConfig()


async def train_colocated_hybrid(
    rollouts: list[Rollout],
    gpus: list[GPU],
    batch_size: int,
    slots_per_gpu: int,
    optimizer: GradientOptimizer,
) -> None:
    """Colocated time-multiplexed training -- every GPU alternates both roles."""
    global METRICS
    METRICS = Metrics(name="train_colocated_hybrid")
    METRICS.extra["chunks_per_flush"] = CONFIG.chunks_per_flush

    recorder = BatchRecorder(optimizer, batch_size)
    max_ready = CONFIG.max_ready if CONFIG.max_ready is not None else batch_size // 2
    METRICS.extra["max_ready"] = max_ready

    cl = _Cluster(rollouts, slots_per_gpu, recorder, CONFIG, max_ready, METRICS)
    await asyncio.gather(*[_gpu_scheduler(gpu, cl) for gpu in gpus])
    await recorder.close()


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from grpo import Config, ModelState, make_rollouts, run_and_report, train_baseline

    async def main() -> None:
        cfg = Config()
        rollouts = make_rollouts(cfg)
        model_state = ModelState()
        gpus = [GPU(id=i, cfg=cfg, model_state=model_state) for i in range(cfg.num_gpus)]
        optimizer = GradientOptimizer(batch_size=cfg.batch_size, model_state=model_state)

        base = await run_and_report("train_baseline", train_baseline, rollouts, gpus, optimizer, cfg)

        for k in (1, 2, 4):
            CONFIG.chunks_per_flush = k
            got = await run_and_report(
                f"colocated_hybrid (chunks_per_flush={k})",
                train_colocated_hybrid,
                rollouts,
                gpus,
                optimizer,
                cfg,
            )
            print(f"  Metrics    : {METRICS.line()}")
            if base and got:
                print(f"  Speedup    : {base / got:.2f}x")

    asyncio.run(main())

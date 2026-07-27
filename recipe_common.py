"""Shared plumbing for the four verl-inspired recipes.

Nothing algorithmic lives here -- just the bits every recipe needs and that
would otherwise be copy-pasted four times: batch assembly, an edge-triggered
broadcast primitive, and the capacity model used to size GPU role splits.

Everything in `grpo.py` below the "DO NOT MODIFY" line is treated as frozen
hardware. These helpers only ever call the public surface: `GPU.run_inference`,
`GPU.run_backward`, `GradientOptimizer.optimizer_step`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from statistics import fmean, median

from grpo import GPU, Config, GradientOptimizer, Rollout


class BatchRecorder:
    """Collects BACKWARDED rollouts and fires `optimizer_step` every batch_size.

    Ordering note: batches are assembled in *backward-completion order*, not in
    rollout id order. That is deliberate and it is what makes the whole pipeline
    legal -- the assignment allows out-of-order processing as long as batches are
    recorded correctly.

    Freshness note: a batch is submitted the instant its Nth member finishes
    backward, so the newest member of every batch is at most one backward-latency
    old. That property is load-bearing; see `docs` in recipe 04 for why.
    """

    def __init__(self, optimizer: GradientOptimizer, batch_size: int, on_step=None) -> None:
        self._opt = optimizer
        self._bs = batch_size
        self._on_step = on_step
        self._pending: list[Rollout] = []
        self._lock = asyncio.Lock()
        self.steps = 0
        self.stale_steps = 0

    async def record(self, rollouts: list[Rollout]) -> None:
        async with self._lock:
            self._pending.extend(rollouts)
            while len(self._pending) >= self._bs:
                batch = self._pending[: self._bs]
                del self._pending[: self._bs]
                before = await self._opt.model_state.get_weights_version()
                await self._opt.optimizer_step(batch)
                after = await self._opt.model_state.get_weights_version()
                if after == before:
                    self.stale_steps += 1
                self.steps += 1
                if self._on_step is not None:
                    await self._on_step(self.steps)

    async def close(self) -> None:
        """Record whatever is left over as the (single) allowed partial batch."""
        async with self._lock:
            if self._pending:
                await self._opt.optimizer_step(self._pending, final_input=True)
                self._pending = []
                self.steps += 1


class Gate:
    """FIFO waiter queue with *counted* wakeups.

    Use this, not `Signal`, whenever the number of waiters can be slot-scale. A
    bounded queue with 128 blocked producers plus a per-item broadcast wakes 128
    tasks per item, 127 of which immediately re-block -- O(waiters) work per
    event, order 1.6M wakeups over a run at this config. That thundering herd is
    invisible on a real cluster (it burns CPU, not GPU) but this simulator
    measures wall-clock, so it lands squarely in the "GPU time" column.

    Waking exactly as many waiters as there are freed resources makes it O(1).
    """

    __slots__ = ("_waiters",)

    def __init__(self) -> None:
        self._waiters: deque[asyncio.Future] = deque()

    def __len__(self) -> int:
        return len(self._waiters)

    def wait(self) -> "asyncio.Future":
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        return fut

    def wake(self, n: int = 1) -> None:
        while n > 0 and self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)
                n -= 1

    def wake_all(self) -> None:
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)


class Signal:
    """One-to-many wakeup with no lost-wakeup race.

    `asyncio.Event` + `clear()` is racy with multiple waiters -- one waiter can
    clear the event before its peers are scheduled. Swapping the event object on
    notify avoids that: waiters bind to the event instance at await time, so a
    notify wakes exactly the set of tasks that were waiting when it fired.

    Correct for a handful of waiters (one per GPU). For slot-scale waiter counts
    use `Gate` instead -- see the note there.
    """

    __slots__ = ("_evt",)

    def __init__(self) -> None:
        self._evt = asyncio.Event()

    def notify(self) -> None:
        evt, self._evt = self._evt, asyncio.Event()
        evt.set()

    async def wait(self) -> None:
        await self._evt.wait()


@dataclass(frozen=True)
class CapacityModel:
    """Steady-state throughput of one GPU in each role, in rollouts/second.

    Inference is *streaming*: `slots` concurrent requests, each taking on average
    `mean_latency`, so a GPU retires slots/mean_latency rollouts per second.

    Backward is *batched and exclusive*: `slots` rollouts per call, one call at a
    time, each call costing `median_latency * backward_forward_ratio` (see
    `backward_latency_for`). So a GPU retires slots/backward_latency per second.

    Both are derived from the actual rollout latencies. In production you would
    get these from the previous epoch's profile; here we have them up front.
    """

    infer_rps: float
    backward_rps: float

    @classmethod
    def from_rollouts(cls, rollouts: list[Rollout], cfg: Config) -> "CapacityModel":
        lat = [r.inference_latency for r in rollouts]
        mean_s = fmean(lat) / 1000.0
        bwd_s = (median(lat) * cfg.backward_forward_ratio) / 1000.0
        return cls(
            infer_rps=cfg.slots_per_gpu / mean_s,
            backward_rps=cfg.slots_per_gpu / bwd_s,
        )

    def plan_split(self, num_gpus: int, total_rollouts: int) -> tuple[int, int]:
        """Pick (gen_gpus, train_gpus) minimising the makespan of the slower role.

        Both roles must process every rollout exactly once, so this reduces to
        balancing total_rollouts/(G*infer_rps) against total_rollouts/(T*bwd_rps).
        Returns at least one GPU per role.
        """
        best, best_cost = (1, num_gpus - 1), float("inf")
        for gen in range(1, num_gpus):
            train = num_gpus - gen
            cost = max(
                total_rollouts / (gen * self.infer_rps),
                total_rollouts / (train * self.backward_rps),
            )
            if cost < best_cost:
                best, best_cost = (gen, train), cost
        return best

    def describe(self, num_gpus: int, total_rollouts: int) -> str:
        gen, train = self.plan_split(num_gpus, total_rollouts)
        return (
            f"infer={self.infer_rps:.0f} rps/gpu  backward={self.backward_rps:.0f} rps/gpu"
            f"  -> split {gen} gen / {train} train"
        )


async def run_backward_and_record(
    gpu: GPU,
    chunk: list[Rollout],
    recorder: BatchRecorder,
) -> None:
    """Backward one chunk, then hand it to batch assembly.

    Kept in one place because the ordering matters: `run_backward` marks the
    rollouts BACKWARDED, and `optimizer_step` asserts that state. Recording
    before the backward returns is the single most common way to fail Check 2.
    """
    await gpu.run_backward(chunk)
    await recorder.record(chunk)


def chunked(items: list[Rollout], size: int) -> list[list[Rollout]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class Metrics:
    """Per-run counters. Printed by bench_all.py; cheap enough to always collect."""

    name: str = ""
    role_switches: int = 0
    backward_calls: int = 0
    drain_s: float = 0.0
    wait_prev_gen_s: float = 0.0
    peak_queue_depth: int = 0
    stale_batches: int = 0
    param_syncs: int = 0
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        bits = []
        if self.role_switches:
            bits.append(f"role_switches={self.role_switches}")
        if self.backward_calls:
            bits.append(f"backward_calls={self.backward_calls}")
        if self.drain_s:
            bits.append(f"drain={self.drain_s * 1000:.0f}ms(sim)")
        if self.wait_prev_gen_s:
            bits.append(f"wait_prev_gen={self.wait_prev_gen_s * 1000:.0f}ms(sim)")
        if self.peak_queue_depth:
            bits.append(f"peak_q={self.peak_queue_depth}")
        if self.param_syncs:
            bits.append(f"param_syncs={self.param_syncs}")
        if self.stale_batches:
            bits.append(f"STALE_BATCHES={self.stale_batches}")
        for k, v in self.extra.items():
            bits.append(f"{k}={v}")
        return "  ".join(bits)

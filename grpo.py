import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from statistics import median


@dataclass
class Config:
    num_gpus: int = 4
    slots_per_gpu: int = 64  # max concurrent operations per GPU
    batch_size: int = 256  # rollouts per optimizer step
    num_batches: int = 50  # total rollouts = num_batches * batch_size
    inference_latency_lo: float = 50.0  # minimum inference latency in ms
    inference_latency_hi: float = 600.0  # maximum inference latency in ms
    inference_beta_alpha: float = 1.5  # beta distribution alpha parameter for sampling latencies
    inference_beta_beta: float = 3.0  # beta distribution beta parameter for sampling latencies
    time_scale: int = 100_000  # simulation speedup factor (higher = faster simulation)
    backward_forward_ratio: float = 1.0  # ratio of backward to forward latency (1.0 = equal time)

    @property
    def num_rollouts(self) -> int:
        return self.num_batches * self.batch_size

    def sim(self, seconds: float) -> float:
        return seconds / self.time_scale

    def rand_inference_latency(self) -> float:
        t = random.betavariate(self.inference_beta_alpha, self.inference_beta_beta)
        return self.inference_latency_lo + t * (self.inference_latency_hi - self.inference_latency_lo)


# ===========================================================================
# TRAINING FUNCTIONS - IMPLEMENT YOUR SOLUTION HERE
# ===========================================================================


"""TODO: Implement an async training pipeline that improves throughput.

Problem:
    Analyze train_baseline to identify inefficiencies that limit GPU utilization.
    Then implement train_async to improve throughput while respecting all constraints.

Key Concepts:
    1. GPU Operations:
       - Inference: GPUs can stream multiple inference requests concurrently (up to slots_per_gpu)
       - Backward: GPUs process backward passes in batches (up to slots_per_gpu rollouts at once)
       - A GPU can ONLY do inference OR backward at one time, never both simultaneously

    2. Weights Version (Optimizer Steps):
       - Each time optimizer_step is called, the model weights improve (version increments)
       - Rollouts capture which weights version they used during inference
       - For good training dynamics, the model should improve continuously throughout training,
         not just at the end (i.e., weights version should increase steadily, not all at once)

Constraints:
    - Each rollout must complete inference before its backward pass
    - Rollouts of size `batch_size` must be recorded together via optimizer_step
    - All rollouts must be processed and recorded exactly once
    - A GPU can only do inference OR backward at a single time (see GPU class docstring)
    - The model should be updated continuously: final weights version should be close to num_batches
      (within 10% tolerance) to ensure the model improves throughout training

Hint:
    Rollouts can be processed out of order, as long as batches are recorded correctly.
"""


async def train_async(
    rollouts: list["Rollout"],
    gpus: list["GPU"],
    batch_size: int,
    slots_per_gpu: int,
    optimizer: "GradientOptimizer",
) -> None:
    """Async training with some GPUS dedicated to inference and some to backward, using queues to pipeline work."""
    pass


async def train_baseline(
    rollouts: list["Rollout"],
    gpus: list["GPU"],
    batch_size: int,
    slots_per_gpu: int,
    optimizer: "GradientOptimizer",
) -> None:
    """baseline training: process one batch at a time (inference -> backward -> optimizer_step).

    Uses all GPUs for inference, then all GPUs for backward."""
    for batch_start in range(0, len(rollouts), batch_size):
        batch = rollouts[batch_start : batch_start + batch_size]

        # Run inference for all rollouts in the batch across all GPUs
        await asyncio.gather(*[gpus[i % len(gpus)].run_inference(r) for i, r in enumerate(batch)])

        # Run backward in chunks, distributing across all GPUs
        per_gpu_chunks = distribute_to_gpus(batch, len(gpus), slots_per_gpu)
        await asyncio.gather(*[run_backward_chunks(gpu, chunks) for gpu, chunks in zip(gpus, per_gpu_chunks)])

        await optimizer.optimizer_step(batch)


def distribute_to_gpus(rollouts: list["Rollout"], num_gpus: int, slots_per_gpu: int) -> list[list[list["Rollout"]]]:
    """Split rollouts into per-GPU chunk lists, interleaving across GPUs.

    Returns a list of length num_gpus, where each element is the ordered
    list of chunks that GPU should process.
    """
    chunks = [rollouts[i : i + slots_per_gpu] for i in range(0, len(rollouts), slots_per_gpu)]
    per_gpu_chunks: list[list[list["Rollout"]]] = [[] for _ in range(num_gpus)]
    for i, chunk in enumerate(chunks):
        per_gpu_chunks[i % num_gpus].append(chunk)
    return per_gpu_chunks


async def run_backward_chunks(gpu: "GPU", chunks: list[list["Rollout"]]) -> None:
    for chunk in chunks:
        await gpu.run_backward(chunk)


# ===========================================================================
# SUPPORT CLASSES AND FUNCTIONS (DO NOT MODIFY)
# ===========================================================================


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


class RolloutState(Enum):
    CREATED = auto()
    INFERENCED = auto()
    BACKWARDED = auto()


@dataclass
class Rollout:
    id: int
    inference_latency: float  # pre-determined
    state: RolloutState = RolloutState.CREATED
    weights_version: int = -1  # Which weights version (optimizer step count) was used for inference

    def mark_inferenced(self, weights_version: int) -> None:
        assert self.state == RolloutState.CREATED, f"Expected CREATED, got {self.state}"
        self.state = RolloutState.INFERENCED
        self.weights_version = weights_version

    def mark_backwarded(self) -> None:
        assert self.state == RolloutState.INFERENCED, f"Expected INFERENCED, got {self.state}"
        self.state = RolloutState.BACKWARDED

    def reset(self) -> None:
        self.state = RolloutState.CREATED
        self.weights_version = -1


def make_rollouts(cfg: Config) -> list["Rollout"]:
    """Create rollouts with pre-determined inference latencies."""
    return [Rollout(id=i, inference_latency=cfg.rand_inference_latency()) for i in range(cfg.num_rollouts)]


def backward_latency_for(rollouts: list["Rollout"], cfg: Config) -> float:
    """Compute backward latency as median of the actual rollouts being backwarded, scaled by backward_forward_ratio"""
    if not rollouts:
        return 0.0
    return median(r.inference_latency for r in rollouts) * cfg.backward_forward_ratio


# ---------------------------------------------------------------------------
# Model state tracking
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    """Shared state tracking weights version (optimizer step count) across GPUs and optimizer.

    Weights version represents how many times the model has been updated via optimizer steps.
    This ensures rollouts use consistent model versions during inference:
    - GPUs capture the current weights version before running inference
    - Rollouts remember which weights version they used
    - Optimizer increments weights version after applying gradients

    This tracks whether the training loop is updating the model continuously (good)
    or batching all updates at the end (bad for learning dynamics).
    """

    _weights_version: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_weights_version(self) -> int:
        """Read current weights version."""
        async with self._lock:
            return self._weights_version

    async def try_increment(self, max_rollout_version: int) -> bool:
        """Increment weights version if max_rollout_version matches current.
        Returns True if incremented, False otherwise."""
        async with self._lock:
            if max_rollout_version >= self._weights_version:
                self._weights_version += 1
                return True
            return False

    def reset(self) -> None:
        """Reset weights version counter to 0."""
        self._weights_version = 0


# ---------------------------------------------------------------------------
# Gradient optimizer
# ---------------------------------------------------------------------------


@dataclass
class GradientOptimizer:
    batch_size: int
    model_state: ModelState
    rollouts: list[Rollout] = field(default_factory=list)
    _finalized: bool = field(default=False, init=False)

    async def optimizer_step(self, rollouts: list[Rollout], final_input: bool = False) -> None:
        """Apply gradients from a batch of rollouts and potentially update weights version."""
        if final_input:
            assert not self._finalized, "final_input=True can only be used once for the final batch"
            self._finalized = True
        if not final_input:
            assert len(rollouts) == self.batch_size, f"Expected batch of {self.batch_size}, got {len(rollouts)}"
        for r in rollouts:
            assert r.state == RolloutState.BACKWARDED, (
                f"Rollout (id={r.id}) recorded before BACKWARDED (state={r.state})"
            )

        # Check max weights version from this batch and try to increment
        max_version = max(r.weights_version for r in rollouts)
        await self.model_state.try_increment(max_version)

        self.rollouts.extend(rollouts)

    def reset(self) -> None:
        self.rollouts.clear()
        self._finalized = False
        self.model_state.reset()

    def summary(self, num_rollouts: int) -> str:
        seen = len(self.rollouts)
        all_backwarded = all(r.state == RolloutState.BACKWARDED for r in self.rollouts)
        return f"recorded={seen}/{num_rollouts}  all_backwarded={all_backwarded}"


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@dataclass
class GPU:
    """Simulates a GPU that can run inference and backward passes.

    The GPU uses a semaphore with `slots_per_gpu` slots to control concurrency:

    Inference (streaming):
      - Can run multiple inferences concurrently, up to slots_per_gpu
      - Each inference request acquires 1 slot while running, then releases it
      - Example: With slots_per_gpu=32, you can call run_inference() 100 times
        concurrently, but only 32 will execute at once; the rest queue up

    Backward (batched):
      - Processes an entire batch of rollouts together (up to slots_per_gpu rollouts)
      - Acquires ALL slots during the backward pass, blocking everything else
      - This means backward operations are "all or nothing" - the GPU is fully
        dedicated to the backward pass until it completes

    Important: A GPU can only do inference OR backward at any given time, never both.
    When a backward pass is running (holding all slots), no inference can run.
    When inference is running (using some slots), backward must wait for all slots.
    """

    id: int
    cfg: Config
    model_state: ModelState
    _sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.cfg.slots_per_gpu)

    @property
    def slots_per_gpu(self):
        return self.cfg.slots_per_gpu

    async def run_inference(self, rollout: Rollout) -> None:
        # Capture current weights version before inference
        async with self._sem:
            await asyncio.sleep(self.cfg.sim(rollout.inference_latency))
        weights_version = await self.model_state.get_weights_version()
        rollout.mark_inferenced(weights_version)

    async def run_backward(self, rollouts: list[Rollout]) -> None:
        assert len(rollouts) <= self.cfg.slots_per_gpu, (
            f"GPU {self.id} can only process {self.cfg.slots_per_gpu} rollouts at once, got {len(rollouts)}"
        )
        latency = backward_latency_for(rollouts, self.cfg)
        for _ in range(self.cfg.slots_per_gpu):
            await self._sem.acquire()
        try:
            await asyncio.sleep(self.cfg.sim(latency))
        finally:
            for _ in range(self.cfg.slots_per_gpu):
                self._sem.release()
        for r in rollouts:
            r.mark_backwarded()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TrainingValidator:
    """Validates training runs to ensure correctness and good training dynamics."""

    @staticmethod
    async def validate(
        optimizer: GradientOptimizer,
        num_rollouts: int,
        num_batches: int,
    ) -> list[str]:
        """Validate that all rollouts were processed correctly and model was updated sufficiently.

        Returns a list of error messages. Empty list means validation passed.
        """
        errors = []

        # Check 1: All rollouts were recorded
        # Why this matters: Missing rollouts means some data was lost or not processed.
        # This violates the requirement that all rollouts must be processed exactly once.
        if len(optimizer.rollouts) != num_rollouts:
            errors.append(
                f"MISSING ROLLOUTS: Expected {num_rollouts} rollouts, got {len(optimizer.rollouts)}. "
                f"This means {num_rollouts - len(optimizer.rollouts)} rollouts were not recorded. "
                f"Every rollout must be processed and recorded exactly once."
            )

        # Check 2: All rollouts completed the full pipeline
        # Why this matters: Rollouts must go through CREATED -> INFERENCED -> BACKWARDED.
        # If rollouts are not in BACKWARDED state, they were recorded prematurely or incorrectly.
        bad_state = [r for r in optimizer.rollouts if r.state != RolloutState.BACKWARDED]
        if bad_state:
            state_counts = {}
            for r in bad_state:
                state_counts[r.state] = state_counts.get(r.state, 0) + 1
            errors.append(
                f"INCOMPLETE ROLLOUTS: {len(bad_state)} rollout(s) not in BACKWARDED state. "
                f"State breakdown: {state_counts}. "
                f"This indicates rollouts were recorded before completing backward pass, "
                f"which violates the required processing order: inference -> backward -> record."
            )

        # Check 3: Model was updated continuously, not all at once
        # Why this matters: In GRPO, the model should be updated after each batch to ensure
        # that later rollouts benefit from earlier training updates. If the weights version
        # is too low, it suggests all optimizer steps were batched at the end, which means
        # the model didn't improve during the rollout collection phase. This is bad for
        # training dynamics because rollouts should be collected with progressively better models.
        final_version = await optimizer.model_state.get_weights_version()
        min_expected = num_batches - 0.1 * num_batches  # Allow 10% leeway for pipeline delays
        if final_version < min_expected:
            errors.append(
                f"INSUFFICIENT MODEL UPDATES: Model only updated {final_version} times, "
                f"expected at least {min_expected:.0f} (out of {num_batches} batches). "
                f"This suggests optimizer steps were delayed and batched near the end of training "
                f"rather than applied continuously. This is problematic because it means most rollouts "
                f"were collected using stale model versions instead of progressively improved models, "
                f"which defeats the purpose of online learning in GRPO."
            )

        return errors


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def run_and_report(
    name: str,
    fn,
    rollouts: list[Rollout],
    gpus: list[GPU],
    optimizer: GradientOptimizer,
    cfg: Config,
) -> float | None:
    """Run a training function and report results. Returns real time in seconds, or None if failed."""
    for r in rollouts:
        r.reset()
    optimizer.reset()

    print("─" * 52)
    print(f"[{name}]\n")
    try:
        t0 = time.perf_counter()
        await fn(rollouts, gpus, cfg.batch_size, cfg.slots_per_gpu, optimizer)
        wall = time.perf_counter() - t0
    except NotImplementedError as e:
        print(f"  skipped: {e}")
        return None

    real_s = wall * cfg.time_scale
    real_m = real_s / 60

    errors = await TrainingValidator.validate(optimizer, cfg.num_rollouts, cfg.num_batches)

    if errors:
        print("\n  Validation failed:\n")
        for e in errors:
            print(f"  !! {e}\n")
        return None

    final_version = await optimizer.model_state.get_weights_version()
    print(f"  Sim  time  : {wall:.3f}s")
    print(f"  Real time  : {real_m:.1f} min  ({real_s:.0f}s)")
    print(f"  Throughput : {cfg.num_rollouts / real_m:.1f} rollouts/min")
    print(f"  Optimizer  : {optimizer.summary(cfg.num_rollouts)}")
    print(f"  Updates    : {final_version}/{cfg.num_batches}")
    return real_s


async def benchmark(cfg: Config) -> None:
    print("=" * 52)
    print("  GRPO simulation")
    print("─" * 52)
    print(f"  GPUs         : {cfg.num_gpus} × {cfg.slots_per_gpu} slots")
    print(f"  Batches      : {cfg.num_batches}")
    print(f"  Batch size   : {cfg.batch_size} rollouts/step")
    print(f"  Rollouts     : {cfg.num_rollouts}")
    print("=" * 52)

    rollouts = make_rollouts(cfg)

    # Create shared model state
    model_state = ModelState()

    # Create GPUs and optimizer with shared model state
    gpus = [GPU(id=i, cfg=cfg, model_state=model_state) for i in range(cfg.num_gpus)]
    optimizer = GradientOptimizer(batch_size=cfg.batch_size, model_state=model_state)

    results: dict[str, float] = {}
    for name, fn in [
        ("train_baseline", train_baseline),
        ("train_async", train_async),
    ]:
        real_s = await run_and_report(name, fn, rollouts, gpus, optimizer, cfg)
        if real_s is not None:
            results[name] = real_s

    # Print comparison if both completed successfully
    if "train_baseline" in results and "train_async" in results:
        seq_time = results["train_baseline"]
        async_time = results["train_async"]
        speedup = seq_time / async_time
        print("─" * 52)
        print(f"  Speedup: {speedup:.2f}x faster than baseline")

    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(benchmark(Config()))

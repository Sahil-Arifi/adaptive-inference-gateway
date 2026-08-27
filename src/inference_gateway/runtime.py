"""Service-facing inference runtime, executor, and operational statistics."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .metrics import GatewayMetrics
from .queueing import ScheduledResult, SchedulerClosedError


class InferenceBackend(Protocol):
    """Structural backend interface needed by the runtime."""

    @property
    def name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def predict_logits(self, batch: NDArray[np.generic]) -> NDArray[np.generic]: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class SchedulerSettings(Protocol):
    @property
    def mode(self) -> object: ...

    @property
    def max_batch_size(self) -> int: ...

    @property
    def max_wait_ms(self) -> float: ...

    @property
    def max_queue_size(self) -> int: ...

    @property
    def request_timeout_ms(self) -> float: ...

    @property
    def inference_workers(self) -> int: ...


@dataclass(frozen=True, slots=True)
class BackendExecution:
    """Result of one blocking backend call measured inside its worker thread."""

    logits: NDArray[np.generic]
    duration_ms: float


class BackendExecutor:
    """Run blocking backend operations away from the HTTP event loop."""

    def __init__(self, backend: InferenceBackend, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.backend = backend
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference-backend",
        )
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def execute(self, batch: NDArray[np.generic]) -> BackendExecution:
        """Execute exactly one ``predict_logits`` call in the dedicated pool."""

        if self._closed:
            raise SchedulerClosedError("backend executor is closed")

        def invoke() -> BackendExecution:
            started = time.perf_counter()
            output = self.backend.predict_logits(batch)
            duration_ms = (time.perf_counter() - started) * 1000.0
            return BackendExecution(logits=np.asarray(output), duration_ms=duration_ms)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, invoke)

    async def warmup(self) -> float:
        """Warm the backend in the same pool used for production inference."""

        if self._closed:
            raise SchedulerClosedError("backend executor is closed")

        def invoke() -> float:
            started = time.perf_counter()
            self.backend.warmup()
            return (time.perf_counter() - started) * 1000.0

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, invoke)

    async def close(self, *, close_backend: bool = True) -> None:
        """Close the backend once, then join all dedicated worker threads."""

        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            backend_error: BaseException | None = None
            # Cancel executor work that has not started, then wait only for
            # calls already running. This prevents direct-mode shutdown from
            # serially executing a large stale work queue.
            await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)
            if close_backend:
                try:
                    await asyncio.to_thread(self.backend.close)
                except BaseException as error:  # shutdown must still join the pool
                    backend_error = error
            if backend_error is not None:
                raise backend_error


class RuntimeStats:
    """Thread-safe cumulative scheduler statistics and stable snapshot math."""

    def __init__(
        self,
        scheduler_mode: str,
        backend: str,
        device: str,
        *,
        started_at: float | None = None,
    ) -> None:
        self.scheduler_mode = scheduler_mode
        self.backend = backend
        self.device = device
        self.started_at = time.monotonic() if started_at is None else started_at
        self._lock = threading.Lock()
        self._accepted_requests = 0
        self._completed_requests = 0
        self._rejected_requests = 0
        self._timed_out_requests = 0
        self._cancelled_requests = 0
        self._failed_requests = 0
        self._backend_inference_calls = 0
        self._batches_executed = 0
        self._current_queue_depth = 0
        self._maximum_observed_queue_depth = 0
        self._total_realized_batch_size = 0
        self._maximum_realized_batch_size = 0
        self._queue_wait_observations = 0
        self._total_queue_wait_ms = 0.0
        self._total_backend_inference_ms = 0.0

    @property
    def accepted_requests(self) -> int:
        with self._lock:
            return self._accepted_requests

    @property
    def completed_requests(self) -> int:
        with self._lock:
            return self._completed_requests

    @property
    def rejected_requests(self) -> int:
        with self._lock:
            return self._rejected_requests

    @property
    def timed_out_requests(self) -> int:
        with self._lock:
            return self._timed_out_requests

    @property
    def cancelled_requests(self) -> int:
        with self._lock:
            return self._cancelled_requests

    @property
    def backend_inference_calls(self) -> int:
        with self._lock:
            return self._backend_inference_calls

    @property
    def batches_executed(self) -> int:
        with self._lock:
            return self._batches_executed

    @property
    def current_queue_depth(self) -> int:
        with self._lock:
            return self._current_queue_depth

    def record_accepted(self, count: int = 1) -> None:
        with self._lock:
            self._accepted_requests += count

    def record_completed(self, count: int = 1) -> None:
        with self._lock:
            self._completed_requests += count

    def record_rejected(self, count: int = 1) -> None:
        with self._lock:
            self._rejected_requests += count

    def record_timeout(self, count: int = 1) -> None:
        with self._lock:
            self._timed_out_requests += count

    def record_cancelled(self, count: int = 1) -> None:
        with self._lock:
            self._cancelled_requests += count

    def record_failed(self, count: int = 1) -> None:
        with self._lock:
            self._failed_requests += count

    def update_queue_depth(self, depth: int) -> None:
        depth = max(depth, 0)
        with self._lock:
            self._current_queue_depth = depth
            self._maximum_observed_queue_depth = max(
                self._maximum_observed_queue_depth,
                depth,
            )

    # Compatibility spelling used by a few callers and tests.
    record_queue_depth = update_queue_depth

    def record_batch(
        self,
        batch_size: int,
        backend_inference_ms: float,
        queue_waits_ms: Sequence[float],
    ) -> None:
        with self._lock:
            self._backend_inference_calls += 1
            self._batches_executed += 1
            self._total_realized_batch_size += batch_size
            self._maximum_realized_batch_size = max(
                self._maximum_realized_batch_size,
                batch_size,
            )
            self._total_backend_inference_ms += max(backend_inference_ms, 0.0)
            self._queue_wait_observations += len(queue_waits_ms)
            self._total_queue_wait_ms += sum(max(value, 0.0) for value in queue_waits_ms)

    def snapshot(self, *, now: float | None = None) -> dict[str, int | float | str]:
        """Return an internally consistent JSON-ready statistics snapshot."""

        observed_at = time.monotonic() if now is None else now
        with self._lock:
            mean_batch_size = (
                self._total_realized_batch_size / self._batches_executed
                if self._batches_executed
                else 0.0
            )
            mean_queue_wait_ms = (
                self._total_queue_wait_ms / self._queue_wait_observations
                if self._queue_wait_observations
                else 0.0
            )
            mean_backend_ms = (
                self._total_backend_inference_ms / self._backend_inference_calls
                if self._backend_inference_calls
                else 0.0
            )
            return {
                "total_accepted_requests": self._accepted_requests,
                "completed_requests": self._completed_requests,
                "rejected_requests": self._rejected_requests,
                "timed_out_requests": self._timed_out_requests,
                "cancelled_requests": self._cancelled_requests,
                "failed_requests": self._failed_requests,
                "backend_inference_calls": self._backend_inference_calls,
                "batches_executed": self._batches_executed,
                "current_queue_depth": self._current_queue_depth,
                "maximum_observed_queue_depth": self._maximum_observed_queue_depth,
                "mean_realized_batch_size": mean_batch_size,
                "maximum_realized_batch_size": self._maximum_realized_batch_size,
                "mean_queue_wait_ms": mean_queue_wait_ms,
                "mean_backend_inference_ms": mean_backend_ms,
                "scheduler_mode": self.scheduler_mode,
                "backend": self.backend,
                "device": self.device,
                "uptime_seconds": max(observed_at - self.started_at, 0.0),
            }

    as_dict = snapshot


class InferenceRuntime:
    """Lifecycle façade used by FastAPI and benchmark servers.

    The runtime owns the dedicated executor and backend lifecycle. Schedulers
    remain event-loop-native and never call a blocking backend method directly.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerSettings,
        *,
        metrics: GatewayMetrics | None = None,
        stats: RuntimeStats | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.metrics = metrics or GatewayMetrics()
        mode = str(config.mode)
        self.stats = stats or RuntimeStats(
            scheduler_mode=mode,
            backend=str(backend.name),
            device=str(backend.device),
        )
        self.executor = BackendExecutor(backend, max_workers=config.inference_workers)

        # Local import avoids a module cycle: scheduler uses the executor and
        # statistics classes above, while the runtime owns scheduler lifecycle.
        from .scheduler import create_scheduler

        self.scheduler = create_scheduler(
            backend,
            config,
            self.stats,
            self.metrics,
            executor=self.executor,
        )
        self.backend_loaded = True
        self.warmup_completed = False
        self._started = False
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()

    @property
    def scheduler_running(self) -> bool:
        return self.scheduler.running

    @property
    def ready(self) -> bool:
        return (
            self.backend_loaded
            and self.warmup_completed
            and self.scheduler_running
            and not self._closed
        )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise SchedulerClosedError("runtime is closed")
            if self._started:
                return
            await self.executor.warmup()
            self.warmup_completed = True
            await self.scheduler.start()
            self._started = True

    async def predict(
        self,
        tensor: NDArray[np.generic],
        *,
        request_id: str | None = None,
        deadline_monotonic: float | None = None,
        timeout_ms: float | None = None,
    ) -> ScheduledResult:
        """Schedule one tensor under an absolute end-to-end deadline."""

        if not self.ready:
            raise SchedulerClosedError("inference runtime is not ready")
        return await self.scheduler.submit(
            request_id or uuid.uuid4().hex,
            np.asarray(tensor),
            deadline_monotonic=deadline_monotonic,
            timeout_ms=timeout_ms,
        )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            scheduler_error: BaseException | None = None
            try:
                await self.scheduler.close()
            except BaseException as error:
                scheduler_error = error
            try:
                await self.executor.close(close_backend=True)
            finally:
                self.backend_loaded = False
                self._started = False
            if scheduler_error is not None:
                raise scheduler_error

    def stats_snapshot(self) -> dict[str, int | float | str]:
        return self.stats.snapshot()

    def metrics_payload(self) -> bytes:
        return self.metrics.render()


# Short name useful for dependency injection in the HTTP service.
Runtime = InferenceRuntime


def as_backend(value: object) -> InferenceBackend:
    """Type-narrow a structurally validated backend for advanced callers."""

    return cast(InferenceBackend, value)


__all__ = [
    "BackendExecution",
    "BackendExecutor",
    "InferenceBackend",
    "InferenceRuntime",
    "Runtime",
    "RuntimeStats",
    "SchedulerSettings",
]

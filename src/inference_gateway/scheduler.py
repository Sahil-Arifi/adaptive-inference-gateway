"""Direct and asynchronous dynamic-microbatch inference schedulers."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .metrics import GatewayMetrics
from .queueing import (
    BoundedRequestQueue,
    DeadlineExceededError,
    PendingRequest,
    QueueFullError,
    RequestTimeoutError,
    ScheduledResult,
    SchedulerClosedError,
    SchedulerError,
)
from .runtime import BackendExecutor, InferenceBackend, RuntimeStats


class SchedulerConfigLike(Protocol):
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


class InferenceScheduler(Protocol):
    mode: str

    @property
    def running(self) -> bool: ...

    @property
    def queue_depth(self) -> int: ...

    async def start(self) -> None: ...

    async def submit(
        self,
        request_id: str,
        tensor: NDArray[np.generic],
        *,
        deadline_monotonic: float | None = None,
        timeout_ms: float | None = None,
    ) -> ScheduledResult: ...

    async def close(self) -> None: ...


def _configured(
    config: SchedulerConfigLike | None,
    name: str,
    override: int | float | None,
    default: int | float,
) -> int | float:
    if override is not None:
        return override
    if config is not None:
        value = getattr(config, name)
        if isinstance(value, (int, float)):
            return value
        raise TypeError(f"scheduler setting {name} must be numeric")
    return default


def _consume_future_exception(future: asyncio.Future[ScheduledResult]) -> None:
    """Prevent abandoned timed-out/shutdown Futures from producing warnings."""

    if future.cancelled():
        return
    try:
        future.exception()
    except asyncio.CancelledError:
        pass


class _BaseScheduler:
    """Shared deadline, executor, statistics, and result-mapping behavior."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfigLike | None,
        stats: RuntimeStats | None,
        metrics: GatewayMetrics | None,
        *,
        mode: str,
        executor: BackendExecutor | None = None,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
        max_queue_size: int | None = None,
        request_timeout_ms: float | None = None,
        inference_workers: int | None = None,
    ) -> None:
        self.backend = backend
        self.mode = mode
        self.max_batch_size = int(_configured(config, "max_batch_size", max_batch_size, 16))
        self.max_wait_ms = float(_configured(config, "max_wait_ms", max_wait_ms, 2.0))
        self.max_queue_size = int(_configured(config, "max_queue_size", max_queue_size, 256))
        self.request_timeout_ms = float(
            _configured(config, "request_timeout_ms", request_timeout_ms, 5000.0)
        )
        self.inference_workers = int(
            _configured(config, "inference_workers", inference_workers, 1)
        )
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if self.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
        if self.inference_workers < 1:
            raise ValueError("inference_workers must be at least 1")

        self.stats = stats or RuntimeStats(
            scheduler_mode=mode,
            backend=str(backend.name),
            device=str(backend.device),
        )
        self.metrics = metrics or GatewayMetrics()
        self._executor = executor or BackendExecutor(backend, self.inference_workers)
        self._owns_executor = executor is None
        self._accepting = False
        self._closed = False

    def _deadline(
        self,
        now: float,
        deadline_monotonic: float | None,
        timeout_ms: float | None,
    ) -> float:
        if deadline_monotonic is not None:
            return deadline_monotonic
        effective_timeout = self.request_timeout_ms if timeout_ms is None else timeout_ms
        if effective_timeout <= 0:
            return now
        return now + (effective_timeout / 1000.0)

    def _new_request(
        self,
        request_id: str,
        tensor: NDArray[np.generic],
        now: float,
        deadline: float,
    ) -> PendingRequest:
        future: asyncio.Future[ScheduledResult] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_future_exception)
        return PendingRequest(
            request_id=request_id,
            tensor=np.asarray(tensor),
            future=future,
            enqueued_at=now,
            absolute_deadline=deadline,
        )

    def _expire(self, request: PendingRequest) -> bool:
        if not request.expire():
            return False
        self.stats.record_timeout()
        self.metrics.record_failure()
        return True

    def _cancel(self, request: PendingRequest) -> bool:
        if not request.cancel():
            return False
        self.stats.record_cancelled()
        self.metrics.record_failure()
        return True

    async def _await_result(self, request: PendingRequest) -> ScheduledResult:
        remaining = request.absolute_deadline - time.monotonic()
        if remaining <= 0:
            self._expire(request)
            raise DeadlineExceededError(f"request {request.request_id} exceeded its deadline")
        try:
            return await asyncio.wait_for(asyncio.shield(request.future), timeout=remaining)
        except TimeoutError as error:
            self._expire(request)
            raise DeadlineExceededError(
                f"request {request.request_id} exceeded its deadline"
            ) from error
        except asyncio.CancelledError:
            self._cancel(request)
            raise

    async def _execute_requests(self, requests: list[PendingRequest]) -> None:
        """Stack once, call the backend once, then fan rows out by input index."""

        now = time.monotonic()
        active: list[PendingRequest] = []
        for request in requests:
            if not request.active:
                continue
            if request.expired(now):
                self._expire(request)
                continue
            active.append(request)
        if not active:
            return

        try:
            batch = np.stack([request.tensor for request in active], axis=0)
        except Exception as error:
            for request in active:
                if request.reject(error):
                    self.stats.record_failed()
                    self.metrics.record_failure()
            return

        execution_started = time.monotonic()
        queue_waits_ms = [
            max((execution_started - request.enqueued_at) * 1000.0, 0.0)
            for request in active
        ]
        for queue_wait_ms in queue_waits_ms:
            self.metrics.observe_queue_wait(queue_wait_ms)

        measured_ms = 0.0
        try:
            execution = await self._executor.execute(batch)
            measured_ms = execution.duration_ms
            logits = np.asarray(execution.logits)
            if logits.ndim < 1 or logits.shape[0] != len(active):
                raise ValueError(
                    "backend output row count does not match the submitted batch: "
                    f"expected {len(active)}, received shape {logits.shape}"
                )
        except Exception as error:
            if measured_ms == 0.0:
                measured_ms = max((time.monotonic() - execution_started) * 1000.0, 0.0)
            self.stats.record_batch(len(active), measured_ms, queue_waits_ms)
            self.metrics.record_backend_call(measured_ms, len(active))
            for request in active:
                if request.reject(error):
                    self.stats.record_failed()
                    self.metrics.record_failure()
            return

        self.stats.record_batch(len(active), measured_ms, queue_waits_ms)
        self.metrics.record_backend_call(measured_ms, len(active))
        completed_at = time.monotonic()
        for index, request in enumerate(active):
            if not request.active:
                continue
            if request.expired(completed_at):
                self._expire(request)
                continue
            result = ScheduledResult(
                request_id=request.request_id,
                logits=np.asarray(logits[index]),
                queue_wait_ms=queue_waits_ms[index],
                backend_inference_ms=measured_ms,
                realized_batch_size=len(active),
            )
            if request.resolve(result):
                self.stats.record_completed()
                self.metrics.observe_request_latency(
                    max((completed_at - request.enqueued_at) * 1000.0, 0.0)
                )

    async def _close_owned_executor(self) -> None:
        if self._owns_executor:
            await self._executor.close(close_backend=True)


class DirectScheduler(_BaseScheduler):
    """Schedule every logical request as its own immediate backend call."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfigLike | None = None,
        stats: RuntimeStats | None = None,
        metrics: GatewayMetrics | None = None,
        *,
        executor: BackendExecutor | None = None,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
        max_queue_size: int | None = None,
        request_timeout_ms: float | None = None,
        inference_workers: int | None = None,
    ) -> None:
        super().__init__(
            backend,
            config,
            stats,
            metrics,
            mode="direct",
            executor=executor,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
            request_timeout_ms=request_timeout_ms,
            inference_workers=inference_workers,
        )
        self._running = False
        self._requests: dict[int, PendingRequest] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def running(self) -> bool:
        return self._running and self._accepting and not self._closed

    @property
    def queue_depth(self) -> int:
        # Direct requests never enter the dynamic request queue.
        return 0

    async def start(self) -> None:
        if self._closed:
            raise SchedulerClosedError("direct scheduler is closed")
        self._accepting = True
        self._running = True
        self.stats.update_queue_depth(0)
        self.metrics.set_queue_depth(0)

    async def submit(
        self,
        request_id: str,
        tensor: NDArray[np.generic],
        *,
        deadline_monotonic: float | None = None,
        timeout_ms: float | None = None,
    ) -> ScheduledResult:
        self.metrics.record_request()
        if not self.running:
            raise SchedulerClosedError("direct scheduler is not running")
        if sum(request.active for request in self._requests.values()) >= self.max_queue_size:
            self.stats.record_rejected()
            self.metrics.record_queue_rejection()
            raise QueueFullError("direct scheduler capacity is full")

        now = time.monotonic()
        deadline = self._deadline(now, deadline_monotonic, timeout_ms)
        request = self._new_request(request_id, tensor, now, deadline)
        self.stats.record_accepted()
        self._requests[id(request)] = request
        if request.expired(now):
            self._expire(request)
        else:
            task = asyncio.create_task(
                self._execute_requests([request]),
                name=f"direct-inference-{request_id}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        try:
            return await self._await_result(request)
        finally:
            self._requests.pop(id(request), None)

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._running = False
        self._closed = True
        close_error = SchedulerClosedError("direct scheduler closed before request completion")
        for request in tuple(self._requests.values()):
            request.reject(close_error)
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self.stats.update_queue_depth(0)
        self.metrics.set_queue_depth(0)
        await self._close_owned_executor()


class DynamicBatchScheduler(_BaseScheduler):
    """Collect compatible requests into real asynchronous microbatches."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfigLike | None = None,
        stats: RuntimeStats | None = None,
        metrics: GatewayMetrics | None = None,
        *,
        executor: BackendExecutor | None = None,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
        max_queue_size: int | None = None,
        request_timeout_ms: float | None = None,
        inference_workers: int | None = None,
    ) -> None:
        super().__init__(
            backend,
            config,
            stats,
            metrics,
            mode="dynamic",
            executor=executor,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
            request_timeout_ms=request_timeout_ms,
            inference_workers=inference_workers,
        )
        self._queue = BoundedRequestQueue(self.max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._active_batch: list[PendingRequest] = []
        self._running = False

    @property
    def running(self) -> bool:
        return (
            self._running
            and self._accepting
            and not self._closed
            and self._worker_task is not None
            and not self._worker_task.done()
        )

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _update_queue_depth(self) -> None:
        depth = self.queue_depth
        self.stats.update_queue_depth(depth)
        self.metrics.set_queue_depth(depth)

    async def start(self) -> None:
        if self._closed:
            raise SchedulerClosedError("dynamic scheduler is closed")
        if self.running:
            return
        self._accepting = True
        self._running = True
        self._worker_task = asyncio.create_task(
            self._batch_worker(),
            name="dynamic-batch-worker",
        )
        self._update_queue_depth()
        # Give the worker one event-loop turn so readiness means it is alive.
        await asyncio.sleep(0)
        if self._worker_task.done():
            await self._worker_task

    async def submit(
        self,
        request_id: str,
        tensor: NDArray[np.generic],
        *,
        deadline_monotonic: float | None = None,
        timeout_ms: float | None = None,
    ) -> ScheduledResult:
        self.metrics.record_request()
        if not self.running:
            raise SchedulerClosedError("dynamic scheduler is not running")
        now = time.monotonic()
        deadline = self._deadline(now, deadline_monotonic, timeout_ms)
        request = self._new_request(request_id, tensor, now, deadline)
        if request.expired(now):
            self.stats.record_accepted()
            self._expire(request)
            return await self._await_result(request)
        try:
            self._queue.put_nowait(request)
        except QueueFullError:
            request.reject(QueueFullError("inference request queue is full"))
            self.stats.record_rejected()
            self.metrics.record_queue_rejection()
            raise
        self.stats.record_accepted()
        self._update_queue_depth()
        return await self._await_result(request)

    async def _first_active_request(self) -> PendingRequest:
        while True:
            request = await self._queue.get()
            self._update_queue_depth()
            if not request.active:
                self._queue.task_done()
                continue
            if request.expired(time.monotonic()):
                self._expire(request)
                self._queue.task_done()
                continue
            return request

    async def _wait_for_next_request(self, wait_seconds: float) -> PendingRequest | None:
        try:
            request = await asyncio.wait_for(self._queue.get(), timeout=wait_seconds)
        except TimeoutError:
            return None
        self._update_queue_depth()
        return request

    async def _collect_batch(self, first: PendingRequest) -> list[PendingRequest]:
        batch = [first]
        collection_end = first.enqueued_at + (self.max_wait_ms / 1000.0)
        while len(batch) < self.max_batch_size:
            # Requests already queued add no collection delay, so drain them up
            # to the batch limit even when this first request waited behind a
            # previous backend call and its nominal window has elapsed.
            try:
                request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                now = time.monotonic()
                remaining = collection_end - now
                if remaining <= 0:
                    break

                # If consuming the rest of the window could itself cross the
                # earliest deadline, flush now and leave time for inference.
                earliest_deadline = min(request.absolute_deadline for request in batch)
                if earliest_deadline - now <= remaining:
                    break
                waited_request = await self._wait_for_next_request(remaining)
                if waited_request is None:
                    break
                request = waited_request
            else:
                self._update_queue_depth()
            if not request.active:
                self._queue.task_done()
                continue
            if request.expired(time.monotonic()):
                self._expire(request)
                self._queue.task_done()
                continue
            if request.shape != first.shape:
                # No await occurs between get and re-admission, so bounded
                # capacity cannot be stolen. Stop this collection window to
                # avoid repeatedly cycling the incompatible head request.
                self._queue.put_nowait(request)
                self._queue.task_done()
                self._update_queue_depth()
                break
            batch.append(request)
        return batch

    async def _batch_worker(self) -> None:
        try:
            while self._accepting:
                first = await self._first_active_request()
                self._active_batch = await self._collect_batch(first)
                await self._execute_requests(self._active_batch)
                for _request in self._active_batch:
                    self._queue.task_done()
                self._active_batch = []
        except asyncio.CancelledError:
            raise
        finally:
            close_error = SchedulerClosedError(
                "dynamic scheduler closed before request completion"
            )
            for request in self._active_batch:
                request.reject(close_error)
                self._queue.task_done()
            self._active_batch = []
            self._queue.reject_all(close_error)
            self._running = False
            self._update_queue_depth()

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting = False
        self._closed = True
        worker = self._worker_task
        if worker is not None and not worker.done():
            worker.cancel()
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        else:
            self._queue.reject_all(
                SchedulerClosedError("dynamic scheduler closed before request completion")
            )
        self._running = False
        self._update_queue_depth()
        await self._close_owned_executor()


def create_scheduler(
    backend: InferenceBackend,
    config: SchedulerConfigLike,
    stats: RuntimeStats | None = None,
    metrics: GatewayMetrics | None = None,
    *,
    executor: BackendExecutor | None = None,
) -> InferenceScheduler:
    """Create the configured scheduler with one shared public contract."""

    raw_mode = getattr(config.mode, "value", config.mode)
    mode = str(raw_mode).lower()
    scheduler_type: type[DirectScheduler] | type[DynamicBatchScheduler]
    if mode == "direct":
        scheduler_type = DirectScheduler
    elif mode == "dynamic":
        scheduler_type = DynamicBatchScheduler
    else:
        raise ValueError(f"unsupported scheduler mode: {raw_mode}")
    return scheduler_type(
        backend,
        config,
        stats,
        metrics,
        executor=executor,
    )


# Concise aliases for callers and tests.
DynamicScheduler = DynamicBatchScheduler
Scheduler = InferenceScheduler


__all__ = [
    "DeadlineExceededError",
    "DirectScheduler",
    "DynamicBatchScheduler",
    "DynamicScheduler",
    "InferenceScheduler",
    "QueueFullError",
    "RequestTimeoutError",
    "ScheduledResult",
    "Scheduler",
    "SchedulerClosedError",
    "SchedulerError",
    "create_scheduler",
]

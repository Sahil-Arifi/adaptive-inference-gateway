"""Bounded request queue primitives used by the inference schedulers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class SchedulerError(RuntimeError):
    """Base class for failures that have an explicit service-level mapping."""


class QueueFullError(SchedulerError):
    """Raised when a scheduler cannot admit another request."""


class DeadlineExceededError(SchedulerError):
    """Raised when inference does not finish before the request deadline."""


class SchedulerClosedError(SchedulerError):
    """Raised when work is submitted to, or interrupted by, a closed scheduler."""


# A descriptive compatibility alias for callers that use timeout terminology.
RequestTimeoutError = DeadlineExceededError


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    """One request's output and scheduler telemetry.

    ``logits`` is one row of the backend's batched output. Keeping the request ID
    in the result makes accidental response reordering observable in tests and in
    higher-level runtime code.
    """

    request_id: str
    logits: NDArray[np.generic]
    queue_wait_ms: float
    backend_inference_ms: float
    realized_batch_size: int


@dataclass(slots=True)
class PendingRequest:
    """A logical inference request owned by the event loop."""

    request_id: str
    tensor: NDArray[np.generic]
    future: asyncio.Future[ScheduledResult]
    enqueued_at: float
    absolute_deadline: float
    cancelled: bool = False
    timed_out: bool = False

    @property
    def enqueue_timestamp(self) -> float:
        """Compatibility spelling used in architecture documentation."""

        return self.enqueued_at

    @property
    def deadline(self) -> float:
        """Return the absolute monotonic deadline."""

        return self.absolute_deadline

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape used to determine batch compatibility."""

        return tuple(self.tensor.shape)

    @property
    def active(self) -> bool:
        """Whether the request can still receive a result."""

        return not self.cancelled and not self.timed_out and not self.future.done()

    def expired(self, now: float) -> bool:
        """Return whether ``now`` is at or beyond the request deadline."""

        return now >= self.absolute_deadline

    def resolve(self, result: ScheduledResult) -> bool:
        """Resolve the request exactly once, returning whether it won the race."""

        if not self.active:
            return False
        self.future.set_result(result)
        return True

    def reject(self, error: BaseException) -> bool:
        """Reject the request exactly once, returning whether it won the race."""

        if not self.active:
            return False
        self.future.set_exception(error)
        return True

    def expire(self) -> bool:
        """Mark the request timed out and settle its Future."""

        if not self.active:
            return False
        self.timed_out = True
        self.future.set_exception(
            DeadlineExceededError(f"request {self.request_id} exceeded its deadline")
        )
        return True

    def cancel(self) -> bool:
        """Mark a disconnected caller cancelled and settle its Future."""

        if not self.active:
            return False
        self.cancelled = True
        self.future.cancel()
        return True


class BoundedRequestQueue:
    """A small typed wrapper around a genuinely bounded ``asyncio.Queue``.

    Admission is intentionally non-blocking. Waiting for queue capacity would
    hide overload and move backpressure into the HTTP event loop; callers instead
    receive ``QueueFullError`` immediately and can map it to HTTP 429.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue(maxsize=max_size)

    @property
    def max_size(self) -> int:
        return self._queue.maxsize

    @property
    def maxsize(self) -> int:
        """Match ``asyncio.Queue.maxsize`` for convenient introspection."""

        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def put_nowait(self, request: PendingRequest) -> None:
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull as error:
            raise QueueFullError("inference request queue is full") from error

    async def get(self) -> PendingRequest:
        return await self._queue.get()

    def get_nowait(self) -> PendingRequest:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def drain(self) -> list[PendingRequest]:
        """Remove every currently queued request without awaiting."""

        requests: list[PendingRequest] = []
        while True:
            try:
                request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return requests
            requests.append(request)
            self._queue.task_done()

    def reject_all(self, error: BaseException) -> int:
        """Drain the queue and reject all still-active requests."""

        return sum(request.reject(error) for request in self.drain())

    def __len__(self) -> int:
        return self.qsize()


def compatible_shapes(requests: Iterable[PendingRequest]) -> bool:
    """Return whether all request tensors can be stacked without reshaping."""

    iterator = iter(requests)
    try:
        first_shape = next(iterator).shape
    except StopIteration:
        return True
    return all(request.shape == first_shape for request in iterator)


# Concise alias retained for callers that prefer the generic name.
RequestQueue = BoundedRequestQueue


__all__ = [
    "BoundedRequestQueue",
    "DeadlineExceededError",
    "PendingRequest",
    "QueueFullError",
    "RequestQueue",
    "RequestTimeoutError",
    "ScheduledResult",
    "SchedulerClosedError",
    "SchedulerError",
    "compatible_shapes",
]

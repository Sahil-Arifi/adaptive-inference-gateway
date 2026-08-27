from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from inference_gateway.queueing import (
    BoundedRequestQueue,
    DeadlineExceededError,
    PendingRequest,
    QueueFullError,
    ScheduledResult,
    compatible_shapes,
)


def make_request(
    request_id: str,
    *,
    shape: tuple[int, ...] = (2, 2),
    deadline_offset: float = 1.0,
) -> PendingRequest:
    now = time.monotonic()
    future: asyncio.Future[ScheduledResult] = asyncio.get_running_loop().create_future()
    return PendingRequest(
        request_id=request_id,
        tensor=np.zeros(shape, dtype=np.float32),
        future=future,
        enqueued_at=now,
        absolute_deadline=now + deadline_offset,
    )


def make_result(request_id: str) -> ScheduledResult:
    return ScheduledResult(
        request_id=request_id,
        logits=np.array([1.0, 2.0], dtype=np.float32),
        queue_wait_ms=0.5,
        backend_inference_ms=1.5,
        realized_batch_size=2,
    )


def test_bounded_queue_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BoundedRequestQueue(0)


@pytest.mark.asyncio
async def test_bounded_queue_enforces_capacity_and_tracks_tasks() -> None:
    queue = BoundedRequestQueue(2)
    first = make_request("first")
    second = make_request("second")
    rejected = make_request("rejected")

    assert queue.max_size == queue.maxsize == 2
    assert queue.empty()
    queue.put_nowait(first)
    queue.put_nowait(second)
    assert len(queue) == queue.qsize() == 2
    assert queue.full()

    with pytest.raises(QueueFullError, match="queue is full"):
        queue.put_nowait(rejected)

    assert await queue.get() is first
    queue.task_done()
    assert queue.get_nowait() is second
    queue.task_done()
    assert queue.empty()
    with pytest.raises(asyncio.QueueEmpty):
        queue.get_nowait()
    await asyncio.wait_for(queue.join(), timeout=0.1)


@pytest.mark.asyncio
async def test_pending_request_resolves_exactly_once() -> None:
    request = make_request("mapped")
    result = make_result("mapped")

    assert request.enqueue_timestamp == request.enqueued_at
    assert request.deadline == request.absolute_deadline
    assert request.shape == (2, 2)
    assert request.active
    assert not request.expired(request.enqueued_at)
    assert request.resolve(result)
    assert await request.future is result
    assert not request.active
    assert not request.resolve(result)
    assert not request.reject(RuntimeError("too late"))
    assert not request.expire()
    assert not request.cancel()


@pytest.mark.asyncio
async def test_pending_request_rejection_timeout_and_cancellation_settle_futures() -> None:
    rejected = make_request("rejected")
    error = ValueError("backend failed")
    assert rejected.reject(error)
    with pytest.raises(ValueError, match="backend failed"):
        await rejected.future

    expired = make_request("expired", deadline_offset=-1.0)
    assert expired.expired(time.monotonic())
    assert expired.expire()
    assert expired.timed_out
    with pytest.raises(DeadlineExceededError, match="expired"):
        await expired.future

    cancelled = make_request("cancelled")
    assert cancelled.cancel()
    assert cancelled.cancelled
    assert cancelled.future.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await cancelled.future


@pytest.mark.asyncio
async def test_drain_and_reject_all_leave_no_unresolved_futures() -> None:
    queue = BoundedRequestQueue(3)
    requests = [make_request(f"request-{index}") for index in range(3)]
    for request in requests:
        queue.put_nowait(request)

    already_done = make_result(requests[0].request_id)
    assert requests[0].resolve(already_done)
    shutdown_error = RuntimeError("shutdown")
    assert queue.reject_all(shutdown_error) == 2
    assert queue.empty()
    await queue.join()

    assert await requests[0].future is already_done
    for request in requests[1:]:
        with pytest.raises(RuntimeError, match="shutdown"):
            await request.future

    assert queue.drain() == []


@pytest.mark.asyncio
async def test_shape_compatibility_handles_empty_matching_and_mismatched_inputs() -> None:
    same = [make_request("one", shape=(3, 4)), make_request("two", shape=(3, 4))]
    different = [same[0], make_request("three", shape=(4, 3))]

    assert compatible_shapes([])
    assert compatible_shapes(same)
    assert not compatible_shapes(different)

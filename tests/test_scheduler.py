from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from inference_gateway.config import SchedulerConfig, SchedulerMode
from inference_gateway.queueing import (
    DeadlineExceededError,
    QueueFullError,
    SchedulerClosedError,
)
from inference_gateway.scheduler import (
    DirectScheduler,
    DynamicBatchScheduler,
    DynamicScheduler,
    create_scheduler,
)


def make_config(
    mode: SchedulerMode = SchedulerMode.DYNAMIC,
    **updates: int | float,
) -> SchedulerConfig:
    values: dict[str, object] = {
        "mode": mode,
        "max_batch_size": 8,
        "max_wait_ms": 10.0,
        "max_queue_size": 64,
        "request_timeout_ms": 2000.0,
        "inference_workers": 1,
    }
    values.update(updates)
    return SchedulerConfig.model_validate(values)


class MappingBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_count = 0
        self._batch_sizes: list[int] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "mapping"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._batch_sizes)

    def _record_call(self, batch_size: int) -> None:
        with self._lock:
            self._call_count += 1
            self._batch_sizes.append(batch_size)

    @staticmethod
    def _logits(batch: np.ndarray) -> np.ndarray:
        identifiers = batch.reshape(batch.shape[0], -1)[:, 0].astype(np.float32)
        return np.stack((identifiers, identifiers + np.float32(100.0)), axis=1)

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self._record_call(int(batch.shape[0]))
        return self._logits(batch)

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class GateBackend(MappingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self.entered.set()
        try:
            if not self.release.wait(timeout=2):
                raise TimeoutError("test did not release the backend")
            return super().predict_logits(batch)
        finally:
            self.exited.set()


class OutcomeBackend(MappingBackend):
    def __init__(self, first_outcome: str) -> None:
        super().__init__()
        self._outcomes = deque((first_outcome, "ok"))

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self._record_call(int(batch.shape[0]))
        outcome = self._outcomes.popleft()
        if outcome == "error":
            raise RuntimeError("backend exploded")
        logits = self._logits(batch)
        if outcome == "wrong_rows":
            return logits[:-1]
        return logits


async def wait_for_thread_event(
    event: threading.Event,
    wait_seconds: float = 1.0,
) -> None:
    assert await asyncio.to_thread(event.wait, wait_seconds), "backend event was not reached"


async def wait_for_queue_depth(
    scheduler: DynamicBatchScheduler,
    expected: int,
) -> None:
    for _ in range(100):
        if scheduler.queue_depth == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"queue depth never reached {expected}; got {scheduler.queue_depth}")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_wait_ms": -1.0}, "max_wait_ms"),
        ({"max_queue_size": 0}, "max_queue_size"),
        ({"request_timeout_ms": 0.0}, "request_timeout_ms"),
        ({"inference_workers": 0}, "inference_workers"),
    ],
)
def test_scheduler_rejects_invalid_direct_overrides(
    override: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectScheduler(MappingBackend(), None, **override)


def test_scheduler_rejects_nonnumeric_protocol_setting() -> None:
    invalid = SimpleNamespace(
        mode="dynamic",
        max_batch_size="eight",
        max_wait_ms=1.0,
        max_queue_size=8,
        request_timeout_ms=1000.0,
        inference_workers=1,
    )
    with pytest.raises(TypeError, match="max_batch_size"):
        DynamicBatchScheduler(MappingBackend(), invalid)


@pytest.mark.asyncio
async def test_factory_selects_modes_and_rejects_unknown_mode() -> None:
    direct = create_scheduler(MappingBackend(), make_config(SchedulerMode.DIRECT))
    dynamic = create_scheduler(MappingBackend(), make_config(SchedulerMode.DYNAMIC))
    assert isinstance(direct, DirectScheduler)
    assert isinstance(dynamic, DynamicBatchScheduler)
    assert DynamicScheduler is DynamicBatchScheduler
    await direct.close()
    await dynamic.close()

    invalid = SimpleNamespace(
        mode="mystery",
        max_batch_size=1,
        max_wait_ms=0.0,
        max_queue_size=1,
        request_timeout_ms=1000.0,
        inference_workers=1,
    )
    with pytest.raises(ValueError, match="unsupported scheduler mode"):
        create_scheduler(MappingBackend(), invalid)


@pytest.mark.asyncio
async def test_direct_scheduler_runs_one_backend_call_per_request_and_maps_rows() -> None:
    backend = MappingBackend()
    scheduler = DirectScheduler(backend, make_config(SchedulerMode.DIRECT))
    with pytest.raises(SchedulerClosedError, match="not running"):
        await scheduler.submit("early", np.ones((2, 2), dtype=np.float32))

    await scheduler.start()
    await scheduler.start()
    results = await asyncio.gather(
        *[
            scheduler.submit(
                f"direct-{index}",
                np.full((2, 2), index, dtype=np.float32),
            )
            for index in range(4)
        ]
    )

    assert scheduler.running
    assert scheduler.mode == "direct"
    assert scheduler.queue_depth == 0
    assert backend.call_count == 4
    assert backend.batch_sizes == (1, 1, 1, 1)
    for index, result in enumerate(results):
        assert result.request_id == f"direct-{index}"
        np.testing.assert_array_equal(
            result.logits,
            np.array([index, index + 100], dtype=np.float32),
        )
        assert result.realized_batch_size == 1
    assert scheduler.stats.snapshot()["mean_realized_batch_size"] == 1.0

    await scheduler.close()
    await scheduler.close()
    assert backend.closed
    assert not scheduler.running
    with pytest.raises(SchedulerClosedError, match="not running"):
        await scheduler.submit("late", np.ones((2, 2), dtype=np.float32))
    with pytest.raises(SchedulerClosedError, match="closed"):
        await scheduler.start()


@pytest.mark.asyncio
async def test_direct_capacity_rejects_and_close_settles_inflight_request() -> None:
    backend = GateBackend()
    scheduler = DirectScheduler(
        backend,
        make_config(SchedulerMode.DIRECT, max_queue_size=1),
    )
    await scheduler.start()
    inflight = asyncio.create_task(
        scheduler.submit("inflight", np.ones((2, 2), dtype=np.float32))
    )
    await wait_for_thread_event(backend.entered)

    with pytest.raises(QueueFullError, match="capacity is full"):
        await scheduler.submit("overflow", np.ones((2, 2), dtype=np.float32))
    assert scheduler.stats.rejected_requests == 1

    close_task = asyncio.create_task(scheduler.close())
    with pytest.raises(SchedulerClosedError, match="closed before request completion"):
        await asyncio.wait_for(inflight, timeout=0.2)
    backend.release.set()
    await close_task
    assert scheduler.stats.snapshot()["current_queue_depth"] == 0


@pytest.mark.asyncio
async def test_max_batch_size_flushes_without_waiting_for_long_window() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=4, max_wait_ms=1000.0),
    )
    await scheduler.start()
    requests = [
        asyncio.create_task(
            scheduler.submit(str(index), np.full((2, 2), index, dtype=np.float32))
        )
        for index in range(4)
    ]
    await wait_for_thread_event(backend.entered, wait_seconds=0.25)
    backend.release.set()
    results = await asyncio.gather(*requests)

    assert backend.batch_sizes == (4,)
    assert all(result.realized_batch_size == 4 for result in results)
    await scheduler.close()


@pytest.mark.asyncio
async def test_wait_window_flushes_a_partial_batch() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=4, max_wait_ms=10.0),
    )
    await scheduler.start()
    request = asyncio.create_task(
        scheduler.submit("single", np.ones((2, 2), dtype=np.float32))
    )

    await wait_for_thread_event(backend.entered)
    backend.release.set()
    result = await request
    assert backend.batch_sizes == (1,)
    assert result.realized_batch_size == 1
    assert result.queue_wait_ms >= 0.0
    await scheduler.close()


@pytest.mark.asyncio
async def test_near_deadline_forces_immediate_partial_batch_flush() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=4, max_wait_ms=1000.0),
    )
    await scheduler.start()
    request = asyncio.create_task(
        scheduler.submit(
            "urgent",
            np.ones((2, 2), dtype=np.float32),
            deadline_monotonic=time.monotonic() + 0.5,
        )
    )

    await wait_for_thread_event(backend.entered, wait_seconds=0.2)
    backend.release.set()
    result = await request
    assert result.request_id == "urgent"
    assert backend.batch_sizes == (1,)
    await scheduler.close()


@pytest.mark.asyncio
async def test_thirty_two_simultaneous_requests_batch_and_map_exactly() -> None:
    backend = MappingBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=8, max_wait_ms=25.0),
    )
    await scheduler.start()
    results = await asyncio.gather(
        *[
            scheduler.submit(
                f"request-{index}",
                np.full((3, 4), index, dtype=np.float32),
            )
            for index in range(32)
        ]
    )

    assert backend.call_count < 32
    assert max(backend.batch_sizes) == 8
    assert sum(backend.batch_sizes) == 32
    for index, result in enumerate(results):
        assert result.request_id == f"request-{index}"
        np.testing.assert_array_equal(
            result.logits,
            np.array([index, index + 100], dtype=np.float32),
        )
    stats = scheduler.stats.snapshot()
    assert stats["completed_requests"] == 32
    assert stats["backend_inference_calls"] == backend.call_count
    assert stats["mean_realized_batch_size"] > 1.0
    assert stats["maximum_realized_batch_size"] == 8
    await scheduler.close()


@pytest.mark.asyncio
async def test_incompatible_shapes_are_executed_in_separate_batches() -> None:
    backend = MappingBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=4, max_wait_ms=5.0),
    )
    await scheduler.start()
    first, second = await asyncio.gather(
        scheduler.submit("first", np.ones((2, 2), dtype=np.float32)),
        scheduler.submit("second", np.full((3, 2), 2.0, dtype=np.float32)),
    )

    assert first.request_id == "first"
    assert second.request_id == "second"
    assert backend.batch_sizes == (1, 1)
    await scheduler.close()


@pytest.mark.asyncio
async def test_dynamic_queue_rejects_immediately_at_capacity() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0, max_queue_size=1),
    )
    await scheduler.start()
    inflight = asyncio.create_task(
        scheduler.submit("inflight", np.ones((2, 2), dtype=np.float32))
    )
    await wait_for_thread_event(backend.entered)
    queued = asyncio.create_task(
        scheduler.submit("queued", np.full((2, 2), 2.0, dtype=np.float32))
    )
    await wait_for_queue_depth(scheduler, 1)

    with pytest.raises(QueueFullError, match="queue is full"):
        await scheduler.submit("overflow", np.ones((2, 2), dtype=np.float32))
    assert scheduler.stats.rejected_requests == 1
    assert "inference_gateway_queue_rejections_total 1.0" in scheduler.metrics.render_text()

    backend.release.set()
    await asyncio.gather(inflight, queued)
    await scheduler.close()


@pytest.mark.asyncio
async def test_queued_request_times_out_while_backend_is_busy() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0),
    )
    await scheduler.start()
    inflight = asyncio.create_task(
        scheduler.submit("inflight", np.ones((2, 2), dtype=np.float32))
    )
    await wait_for_thread_event(backend.entered)

    with pytest.raises(DeadlineExceededError, match="queued"):
        await scheduler.submit(
            "queued",
            np.full((2, 2), 2.0, dtype=np.float32),
            timeout_ms=20.0,
        )
    assert scheduler.stats.timed_out_requests == 1

    backend.release.set()
    await inflight
    await scheduler.close()


@pytest.mark.asyncio
async def test_inflight_request_times_out_without_corrupting_worker() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0),
    )
    await scheduler.start()
    request = asyncio.create_task(
        scheduler.submit(
            "inflight-timeout",
            np.ones((2, 2), dtype=np.float32),
            timeout_ms=20.0,
        )
    )
    await wait_for_thread_event(backend.entered)
    with pytest.raises(DeadlineExceededError, match="inflight-timeout"):
        await request
    assert scheduler.stats.timed_out_requests == 1

    backend.release.set()
    await wait_for_thread_event(backend.exited)
    for _ in range(100):
        if scheduler.stats.backend_inference_calls == 1:
            break
        await asyncio.sleep(0)
    await scheduler.close()
    assert scheduler.stats.backend_inference_calls == 1


@pytest.mark.asyncio
async def test_cancelled_queued_request_does_not_corrupt_later_batches() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0),
    )
    await scheduler.start()
    first = asyncio.create_task(
        scheduler.submit("first", np.ones((2, 2), dtype=np.float32))
    )
    await wait_for_thread_event(backend.entered)
    cancelled = asyncio.create_task(
        scheduler.submit("cancelled", np.full((2, 2), 2.0, dtype=np.float32))
    )
    await wait_for_queue_depth(scheduler, 1)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert scheduler.stats.cancelled_requests == 1

    backend.release.set()
    await first
    later = await scheduler.submit("later", np.full((2, 2), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(later.logits, np.array([3.0, 103.0], dtype=np.float32))
    assert scheduler.stats.completed_requests == 2
    await scheduler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "error_type", "message"),
    [
        ("error", RuntimeError, "backend exploded"),
        ("wrong_rows", ValueError, "output row count"),
    ],
)
async def test_backend_failure_is_propagated_and_worker_recovers(
    outcome: str,
    error_type: type[Exception],
    message: str,
) -> None:
    backend = OutcomeBackend(outcome)
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0),
    )
    await scheduler.start()

    with pytest.raises(error_type, match=message):
        await scheduler.submit("fails", np.ones((2, 2), dtype=np.float32))
    recovered = await scheduler.submit(
        "recovers",
        np.full((2, 2), 4.0, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        recovered.logits,
        np.array([4.0, 104.0], dtype=np.float32),
    )
    assert scheduler.running
    assert backend.call_count == 2
    snapshot = scheduler.stats.snapshot()
    assert snapshot["failed_requests"] == 1
    assert snapshot["completed_requests"] == 1
    assert snapshot["backend_inference_calls"] == 2
    await scheduler.close()


@pytest.mark.asyncio
async def test_shutdown_settles_inflight_and_queued_futures() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0, max_queue_size=4),
    )
    await scheduler.start()
    requests = [
        asyncio.create_task(
            scheduler.submit(str(index), np.full((2, 2), index, dtype=np.float32))
        )
        for index in range(3)
    ]
    await wait_for_thread_event(backend.entered)
    await wait_for_queue_depth(scheduler, 2)

    close_task = asyncio.create_task(scheduler.close())
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*requests, return_exceptions=True),
            timeout=0.2,
        )
    finally:
        backend.release.set()
        await close_task

    assert all(isinstance(result, SchedulerClosedError) for result in results)
    assert all(request.done() for request in requests)
    assert scheduler.queue_depth == 0
    assert scheduler.stats.current_queue_depth == 0
    assert not scheduler.running
    await scheduler.close()
    with pytest.raises(SchedulerClosedError, match="not running"):
        await scheduler.submit("after-close", np.ones((2, 2), dtype=np.float32))
    with pytest.raises(SchedulerClosedError, match="closed"):
        await scheduler.start()


@pytest.mark.asyncio
async def test_blocking_backend_does_not_freeze_scheduler_event_loop() -> None:
    backend = GateBackend()
    scheduler = DynamicBatchScheduler(
        backend,
        make_config(max_batch_size=1, max_wait_ms=0.0),
    )
    await scheduler.start()
    request = asyncio.create_task(
        scheduler.submit("responsive", np.ones((2, 2), dtype=np.float32))
    )
    await wait_for_thread_event(backend.entered)

    heartbeat = asyncio.Event()

    async def pulse() -> None:
        await asyncio.sleep(0)
        heartbeat.set()

    await asyncio.wait_for(pulse(), timeout=0.1)
    assert heartbeat.is_set()
    backend.release.set()
    await request
    await scheduler.close()

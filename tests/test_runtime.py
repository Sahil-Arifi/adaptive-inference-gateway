from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from inference_gateway.backends.fake_backend import FakeBackend
from inference_gateway.config import SchedulerConfig, SchedulerMode
from inference_gateway.queueing import SchedulerClosedError
from inference_gateway.runtime import (
    BackendExecutor,
    InferenceRuntime,
    Runtime,
    RuntimeStats,
    as_backend,
)


def scheduler_config(mode: SchedulerMode = SchedulerMode.DIRECT) -> SchedulerConfig:
    return SchedulerConfig(
        mode=mode,
        max_batch_size=4,
        max_wait_ms=2.0,
        max_queue_size=16,
        request_timeout_ms=1000.0,
        inference_workers=1,
    )


class BlockingBackend:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.warmed_up = False
        self.closed = False
        self.predict_thread_id: int | None = None

    @property
    def name(self) -> str:
        return "blocking"

    @property
    def device(self) -> str:
        return "cpu"

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self.predict_thread_id = threading.get_ident()
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test backend was not released")
        values = batch.reshape(batch.shape[0], -1)[:, :1]
        return np.concatenate((values, values + 1.0), axis=1).astype(np.float32)

    def warmup(self) -> None:
        self.warmed_up = True

    def close(self) -> None:
        self.closed = True


class WarmupFailureBackend(FakeBackend):
    def warmup(self) -> None:
        raise RuntimeError("warmup failed")


class CloseFailureBackend(FakeBackend):
    def close(self) -> None:
        super().close()
        raise RuntimeError("close failed")


def test_runtime_stats_snapshot_math_and_maxima() -> None:
    stats = RuntimeStats("dynamic", "fake", "cpu", started_at=100.0)

    stats.record_accepted(5)
    stats.record_completed(3)
    stats.record_rejected(1)
    stats.record_timeout(2)
    stats.record_cancelled(1)
    stats.record_failed(4)
    stats.update_queue_depth(2)
    stats.record_queue_depth(7)
    stats.update_queue_depth(-1)
    stats.record_batch(4, 10.0, [2.0, 4.0, 6.0, 8.0])
    stats.record_batch(2, 20.0, [10.0, 12.0])

    snapshot = stats.snapshot(now=110.0)
    assert stats.accepted_requests == 5
    assert stats.completed_requests == 3
    assert stats.rejected_requests == 1
    assert stats.timed_out_requests == 2
    assert stats.cancelled_requests == 1
    assert stats.backend_inference_calls == 2
    assert stats.batches_executed == 2
    assert stats.current_queue_depth == 0
    assert snapshot == stats.as_dict(now=110.0)
    assert snapshot["failed_requests"] == 4
    assert snapshot["maximum_observed_queue_depth"] == 7
    assert snapshot["mean_realized_batch_size"] == pytest.approx(3.0)
    assert snapshot["maximum_realized_batch_size"] == 4
    assert snapshot["mean_queue_wait_ms"] == pytest.approx(7.0)
    assert snapshot["mean_backend_inference_ms"] == pytest.approx(15.0)
    assert snapshot["scheduler_mode"] == "dynamic"
    assert snapshot["backend"] == "fake"
    assert snapshot["device"] == "cpu"
    assert snapshot["uptime_seconds"] == pytest.approx(10.0)


def test_empty_runtime_stats_have_zero_means_and_nonnegative_uptime() -> None:
    snapshot = RuntimeStats("direct", "fake", "cpu", started_at=10.0).snapshot(now=5.0)

    assert snapshot["mean_realized_batch_size"] == 0.0
    assert snapshot["mean_queue_wait_ms"] == 0.0
    assert snapshot["mean_backend_inference_ms"] == 0.0
    assert snapshot["uptime_seconds"] == 0.0


@pytest.mark.asyncio
async def test_backend_executor_keeps_event_loop_responsive() -> None:
    backend = BlockingBackend()
    executor = BackendExecutor(backend, max_workers=1)

    assert await executor.warmup() >= 0.0
    assert backend.warmed_up
    submitted_at = time.monotonic()
    inference = asyncio.create_task(executor.execute(np.ones((1, 2), dtype=np.float32)))
    assert await asyncio.to_thread(backend.entered.wait, 1.0)

    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
    assert backend.predict_thread_id != threading.get_ident()

    backend.release.set()
    execution = await inference
    np.testing.assert_array_equal(execution.logits, np.array([[1.0, 2.0]], dtype=np.float32))
    assert execution.duration_ms >= 0.0
    assert submitted_at <= execution.started_at_monotonic <= time.monotonic()

    await executor.close()
    await executor.close()
    assert executor.closed
    assert backend.closed
    with pytest.raises(SchedulerClosedError, match="executor is closed"):
        await executor.execute(np.ones((1, 2), dtype=np.float32))
    with pytest.raises(SchedulerClosedError, match="executor is closed"):
        await executor.warmup()


def test_backend_executor_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BackendExecutor(FakeBackend(), max_workers=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [SchedulerMode.DIRECT, SchedulerMode.DYNAMIC])
async def test_runtime_lifecycle_prediction_stats_and_metrics(mode: SchedulerMode) -> None:
    backend = FakeBackend(num_classes=5)
    runtime = InferenceRuntime(backend, scheduler_config(mode))
    assert Runtime is InferenceRuntime
    assert as_backend(backend) is backend
    assert runtime.backend_loaded
    assert not runtime.ready

    with pytest.raises(SchedulerClosedError, match="not ready"):
        await runtime.predict(np.ones((2, 2), dtype=np.float32))

    await runtime.start()
    await runtime.start()
    assert runtime.ready
    assert runtime.scheduler_running
    assert runtime.warmup_completed
    assert backend.warmed_up

    generated_id = await runtime.predict(np.ones((2, 2), dtype=np.float32))
    explicit_id = await runtime.predict(
        np.full((2, 2), 2.0, dtype=np.float32),
        request_id="explicit",
        timeout_ms=500.0,
    )
    assert len(generated_id.request_id) == 32
    assert explicit_id.request_id == "explicit"
    assert explicit_id.logits.shape == (5,)
    assert runtime.stats_snapshot()["completed_requests"] == 2
    assert b"inference_gateway_backend_calls_total" in runtime.metrics_payload()

    await runtime.close()
    await runtime.close()
    assert not runtime.ready
    assert not runtime.backend_loaded
    assert not runtime.scheduler_running
    with pytest.raises(SchedulerClosedError, match="not ready"):
        await runtime.predict(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(SchedulerClosedError, match="runtime is closed"):
        await runtime.start()


@pytest.mark.asyncio
async def test_runtime_warmup_failure_never_becomes_ready() -> None:
    runtime = InferenceRuntime(WarmupFailureBackend(), scheduler_config())

    with pytest.raises(RuntimeError, match="warmup failed"):
        await runtime.start()
    assert not runtime.ready
    assert not runtime.warmup_completed
    await runtime.close()


@pytest.mark.asyncio
async def test_backend_close_error_is_propagated_after_executor_is_closed() -> None:
    backend = CloseFailureBackend()
    executor = BackendExecutor(backend)

    with pytest.raises(RuntimeError, match="close failed"):
        await executor.close()
    assert executor.closed
    await executor.close()

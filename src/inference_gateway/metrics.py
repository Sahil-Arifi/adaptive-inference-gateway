"""Prometheus instrumentation with deliberately bounded label cardinality."""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

_LATENCY_BUCKETS: Final = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
_QUEUE_WAIT_BUCKETS: Final = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)
_BATCH_SIZE_BUCKETS: Final = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


class GatewayMetrics:
    """Own an isolated Prometheus registry for one gateway instance.

    The runtime has one configured backend, device, and scheduler mode, so none
    of those values need to be labels. Every collector consequently has zero
    user-controlled labels. This makes request IDs and other high-cardinality
    values impossible to add accidentally.
    """

    content_type = CONTENT_TYPE_LATEST

    def __init__(
        self,
        *,
        enabled: bool = True,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.enabled = enabled
        self.registry = registry or CollectorRegistry(auto_describe=True)

        self.request_count: Counter | None = None
        self.request_failure_count: Counter | None = None
        self.queue_rejection_count: Counter | None = None
        self.queue_depth: Gauge | None = None
        self.request_latency: Histogram | None = None
        self.queue_wait: Histogram | None = None
        self.backend_inference_duration: Histogram | None = None
        self.batch_size: Histogram | None = None
        self.backend_call_count: Counter | None = None

        if not enabled:
            return

        self.request_count = Counter(
            "inference_gateway_requests_total",
            "Logical inference requests received by the scheduler.",
            registry=self.registry,
        )
        self.request_failure_count = Counter(
            "inference_gateway_request_failures_total",
            "Logical requests that ended in rejection, timeout, cancellation, or failure.",
            registry=self.registry,
        )
        self.queue_rejection_count = Counter(
            "inference_gateway_queue_rejections_total",
            "Requests rejected immediately because bounded capacity was exhausted.",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "inference_gateway_queue_depth",
            "Current number of requests waiting in the dynamic scheduler queue.",
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "inference_gateway_request_latency_seconds",
            "Scheduler admission-to-completion latency for successful requests.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "inference_gateway_queue_wait_seconds",
            "Time accepted requests spent waiting before backend execution.",
            buckets=_QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.backend_inference_duration = Histogram(
            "inference_gateway_backend_inference_duration_seconds",
            "Wall time of one batched backend invocation.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "inference_gateway_batch_size",
            "Realized logical request count per backend call.",
            buckets=_BATCH_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.backend_call_count = Counter(
            "inference_gateway_backend_calls_total",
            "Batched backend inference calls attempted.",
            registry=self.registry,
        )

    def record_request(self) -> None:
        if self.request_count is not None:
            self.request_count.inc()

    def record_failure(self) -> None:
        if self.request_failure_count is not None:
            self.request_failure_count.inc()

    def record_queue_rejection(self) -> None:
        if self.queue_rejection_count is not None:
            self.queue_rejection_count.inc()
        self.record_failure()

    def set_queue_depth(self, depth: int) -> None:
        if self.queue_depth is not None:
            self.queue_depth.set(max(depth, 0))

    def observe_request_latency(self, latency_ms: float) -> None:
        if self.request_latency is not None:
            self.request_latency.observe(max(latency_ms, 0.0) / 1000.0)

    def observe_queue_wait(self, queue_wait_ms: float) -> None:
        if self.queue_wait is not None:
            self.queue_wait.observe(max(queue_wait_ms, 0.0) / 1000.0)

    def record_backend_call(self, duration_ms: float, realized_batch_size: int) -> None:
        if self.backend_call_count is not None:
            self.backend_call_count.inc()
        if self.backend_inference_duration is not None:
            self.backend_inference_duration.observe(max(duration_ms, 0.0) / 1000.0)
        if self.batch_size is not None:
            self.batch_size.observe(max(realized_batch_size, 1))

    def render(self) -> bytes:
        """Render this instance's registry in Prometheus exposition format."""

        return generate_latest(self.registry)

    def render_text(self) -> str:
        return self.render().decode("utf-8")


# Explicit name used in documentation and a short compatibility alias.
PrometheusMetrics = GatewayMetrics
Metrics = GatewayMetrics


__all__ = ["GatewayMetrics", "Metrics", "PrometheusMetrics"]

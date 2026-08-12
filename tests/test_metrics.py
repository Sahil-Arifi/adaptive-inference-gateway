from __future__ import annotations

from prometheus_client import CollectorRegistry

from inference_gateway.metrics import GatewayMetrics, Metrics, PrometheusMetrics


def test_metrics_instances_are_isolated_and_have_no_dynamic_labels() -> None:
    first = GatewayMetrics()
    second = GatewayMetrics()

    first.record_request()
    first.record_request()
    first.record_failure()
    first.record_queue_rejection()
    first.set_queue_depth(7)
    first.observe_request_latency(250.0)
    first.observe_queue_wait(5.0)
    first.record_backend_call(20.0, 4)

    first_text = first.render_text()
    second_text = second.render_text()
    assert "inference_gateway_requests_total 2.0" in first_text
    assert "inference_gateway_request_failures_total 2.0" in first_text
    assert "inference_gateway_queue_rejections_total 1.0" in first_text
    assert "inference_gateway_queue_depth 7.0" in first_text
    assert "inference_gateway_request_latency_seconds_count 1.0" in first_text
    assert "inference_gateway_request_latency_seconds_sum 0.25" in first_text
    assert "inference_gateway_queue_wait_seconds_sum 0.005" in first_text
    assert "inference_gateway_backend_inference_duration_seconds_sum 0.02" in first_text
    assert "inference_gateway_batch_size_sum 4.0" in first_text
    assert "inference_gateway_backend_calls_total 1.0" in first_text
    assert "request_id" not in first_text
    assert "inference_gateway_requests_total 0.0" in second_text
    assert "inference_gateway_queue_depth 0.0" in second_text


def test_metrics_clamp_negative_observations_and_render_bytes() -> None:
    metrics = GatewayMetrics(registry=CollectorRegistry())

    metrics.set_queue_depth(-5)
    metrics.observe_request_latency(-10.0)
    metrics.observe_queue_wait(-20.0)
    metrics.record_backend_call(-30.0, 0)

    payload = metrics.render()
    text = payload.decode("utf-8")
    assert isinstance(payload, bytes)
    assert "text/plain" in metrics.content_type
    assert "inference_gateway_queue_depth 0.0" in text
    assert "inference_gateway_request_latency_seconds_sum 0.0" in text
    assert "inference_gateway_queue_wait_seconds_sum 0.0" in text
    assert "inference_gateway_backend_inference_duration_seconds_sum 0.0" in text
    assert "inference_gateway_batch_size_sum 1.0" in text


def test_disabled_metrics_are_safe_noops() -> None:
    metrics = GatewayMetrics(enabled=False)

    metrics.record_request()
    metrics.record_failure()
    metrics.record_queue_rejection()
    metrics.set_queue_depth(10)
    metrics.observe_request_latency(10.0)
    metrics.observe_queue_wait(10.0)
    metrics.record_backend_call(10.0, 8)

    assert metrics.render() == b""
    assert metrics.render_text() == ""


def test_public_metric_aliases_are_stable() -> None:
    assert Metrics is GatewayMetrics
    assert PrometheusMetrics is GatewayMetrics

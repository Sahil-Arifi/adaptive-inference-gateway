"""Data-derived benchmark serialization, Markdown reporting, and charts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from pydantic import ValidationError

matplotlib.use("Agg", force=True)


README_RESULTS_START = "<!-- BENCHMARK_RESULTS_START -->"
README_RESULTS_END = "<!-- BENCHMARK_RESULTS_END -->"
BENCHMARK_RESULTS_SCHEMA_VERSION = 1

_PROVENANCE_ALGORITHM = "sha256-canonical-json-v1"
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "environment",
        "benchmark_config",
        "parity_artifact",
        "image_sequence",
        "cases",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "algorithm",
        "payload_sha256",
        "benchmark_config_sha256",
        "parity_artifact_sha256",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "backend",
        "device",
        "scheduler_mode",
        "concurrency",
        "max_batch_size",
        "max_wait_ms",
        "requests",
        "successful_requests",
        "failures",
        "rejections",
        "timeouts",
        "throughput_requests_per_second",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_queue_wait_ms",
        "p95_queue_wait_ms",
        "mean_backend_inference_ms",
        "mean_realized_batch_size",
        "maximum_realized_batch_size",
        "backend_inference_call_count",
        "batches_executed",
        "duration_seconds",
        "server_stats_delta",
        "server_stats_before",
        "server_stats_after",
        "samples",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "sequence",
        "image_index",
        "elapsed_seconds",
        "status_code",
        "success",
        "timed_out",
        "error",
        "request_id",
        "queue_wait_ms",
        "backend_inference_ms",
        "realized_batch_size",
    }
)
_STATS_COUNTER_FIELDS = frozenset(
    {
        "total_accepted_requests",
        "completed_requests",
        "rejected_requests",
        "timed_out_requests",
        "cancelled_requests",
        "failed_requests",
        "backend_inference_calls",
        "batches_executed",
        "current_queue_depth",
        "maximum_observed_queue_depth",
        "maximum_realized_batch_size",
    }
)
_STATS_MEAN_FIELDS = frozenset(
    {
        "mean_realized_batch_size",
        "mean_queue_wait_ms",
        "mean_backend_inference_ms",
        "uptime_seconds",
    }
)
_STATS_TEXT_FIELDS = frozenset({"scheduler_mode", "backend", "device"})
_STATS_FIELDS = _STATS_COUNTER_FIELDS | _STATS_MEAN_FIELDS | _STATS_TEXT_FIELDS
_DELTA_FIELDS = frozenset(
    {
        "total_accepted_requests",
        "completed_requests",
        "rejected_requests",
        "timed_out_requests",
        "cancelled_requests",
        "failed_requests",
        "backend_inference_calls",
        "batches_executed",
        "current_queue_depth",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "operating_system",
        "system",
        "release",
        "machine",
        "processor",
        "logical_cpu_count",
        "python_version",
        "torch_version",
        "torchvision_version",
        "onnx_version",
        "onnxruntime_version",
        "onnxruntime_available_providers",
        "cuda_available",
        "cuda_version",
        "cuda_device_name",
    }
)
_IMAGE_FIELDS = frozenset(
    {"sequence_index", "filename", "media_type", "byte_length", "sha256"}
)
_PARITY_FIELDS = frozenset(
    {
        "schema_version",
        "model_name",
        "onnx_path",
        "onnx_sha256",
        "input_name",
        "output_name",
        "input_shape",
        "batch_sizes",
        "rtol",
        "atol",
        "seed",
        "device",
        "per_batch",
        "max_abs_difference",
        "mean_abs_difference",
        "top1_agreement",
        "passed",
    }
)
_PARITY_BATCH_FIELDS = frozenset(
    {
        "batch_size",
        "logit_count",
        "max_abs_difference",
        "mean_abs_difference",
        "top1_agreement",
        "allclose",
        "passed",
    }
)

_CSV_FIELDS = (
    "case_id",
    "backend",
    "device",
    "scheduler_mode",
    "concurrency",
    "max_batch_size",
    "max_wait_ms",
    "requests",
    "successful_requests",
    "failures",
    "rejections",
    "timeouts",
    "throughput_requests_per_second",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_queue_wait_ms",
    "p95_queue_wait_ms",
    "mean_backend_inference_ms",
    "mean_realized_batch_size",
    "maximum_realized_batch_size",
    "backend_inference_call_count",
    "batches_executed",
    "duration_seconds",
)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Paths written from one canonical results payload."""

    results_json: Path
    results_csv: Path
    report_markdown: Path
    throughput_vs_p95_chart: Path
    batch_efficiency_chart: Path


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} fields are invalid; missing={missing}, extra={extra}")


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{context} must be a finite number >= {minimum}")
    return number


def _optional_finite(value: Any, context: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, context)


def _strict_text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _strict_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _sha256_digest(value: Any) -> str:
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("benchmark results must contain only finite JSON values") from error
    return hashlib.sha256(canonical).hexdigest()


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = _object(value, "environment")
    _exact_fields(environment, _ENVIRONMENT_FIELDS, "environment")
    for key in ("operating_system", "system", "release", "machine", "processor"):
        _strict_text(environment[key], f"environment.{key}", allow_empty=True)
    _strict_text(environment["python_version"], "environment.python_version")
    logical_cpus = environment["logical_cpu_count"]
    if logical_cpus is not None:
        _strict_int(logical_cpus, "environment.logical_cpu_count", minimum=1)
    for key in (
        "torch_version",
        "torchvision_version",
        "onnx_version",
        "onnxruntime_version",
        "cuda_version",
        "cuda_device_name",
    ):
        if environment[key] is not None:
            _strict_text(environment[key], f"environment.{key}", allow_empty=True)
    providers = environment["onnxruntime_available_providers"]
    if not isinstance(providers, list):
        raise ValueError("environment.onnxruntime_available_providers must be a list")
    for index, provider in enumerate(providers):
        _strict_text(provider, f"environment.onnxruntime_available_providers[{index}]")
    _strict_bool(environment["cuda_available"], "environment.cuda_available")
    return environment


def _validate_config(value: Any) -> Any:
    from inference_gateway.config import GatewaySettings

    config = _object(value, "benchmark_config")
    try:
        settings = GatewaySettings.model_validate(config)
    except ValidationError as error:
        raise ValueError("benchmark_config is invalid") from error
    if settings.model_dump(mode="json") != config:
        raise ValueError("benchmark_config must contain the complete canonical configuration")
    if settings.model.device.value not in {"cpu", "cuda"}:
        raise ValueError("benchmark_config.model.device must explicitly be cpu or cuda")
    return settings


def _validate_parity(value: Any, settings: Any) -> dict[str, Any]:
    from inference_gateway.parity import (
        ParityReport,
        ParityVerificationError,
        validate_production_parity_policy,
    )

    parity = _object(value, "parity_artifact")
    _exact_fields(parity, _PARITY_FIELDS, "parity_artifact")
    batches = parity.get("per_batch")
    if not isinstance(batches, list):
        raise ValueError("parity_artifact.per_batch must be a list")
    for index, batch_value in enumerate(batches):
        batch = _object(batch_value, f"parity_artifact.per_batch[{index}]")
        _exact_fields(
            batch,
            _PARITY_BATCH_FIELDS,
            f"parity_artifact.per_batch[{index}]",
        )
    try:
        report = ParityReport.from_dict(parity)
        validate_production_parity_policy(
            report,
            expected_device=settings.model.device.value,
        )
    except (KeyError, TypeError, ValueError, ParityVerificationError) as error:
        raise ValueError("parity_artifact is invalid") from error
    if Path(report.onnx_path) != settings.model.onnx_path:
        raise ValueError("parity_artifact ONNX path does not match benchmark_config")
    return parity


def _validate_images(value: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"image_sequence must contain exactly {expected_count} images")
    images: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        image = _object(raw, f"image_sequence[{index}]")
        _exact_fields(image, _IMAGE_FIELDS, f"image_sequence[{index}]")
        if _strict_int(image["sequence_index"], f"image_sequence[{index}].sequence_index") != index:
            raise ValueError("image_sequence indices must be contiguous and ordered")
        _strict_text(image["filename"], f"image_sequence[{index}].filename")
        expected_media_type = "image/png" if index % 2 == 0 else "image/jpeg"
        if image["media_type"] != expected_media_type:
            raise ValueError("image_sequence must alternate PNG and JPEG inputs")
        _strict_int(image["byte_length"], f"image_sequence[{index}].byte_length", minimum=1)
        digest = _strict_text(image["sha256"], f"image_sequence[{index}].sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"image_sequence[{index}].sha256 must be lowercase SHA-256")
        images.append(image)
    return images


def _validate_stats(value: Any, context: str) -> dict[str, Any]:
    stats = _object(value, context)
    _exact_fields(stats, _STATS_FIELDS, context)
    for key in _STATS_COUNTER_FIELDS:
        _strict_int(stats[key], f"{context}.{key}")
    for key in _STATS_MEAN_FIELDS:
        _finite_number(stats[key], f"{context}.{key}")
    for key in _STATS_TEXT_FIELDS:
        _strict_text(stats[key], f"{context}.{key}")
    return stats


def _validate_delta(
    value: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    delta = _object(value, "server_stats_delta")
    _exact_fields(delta, _DELTA_FIELDS, "server_stats_delta")
    for key in _DELTA_FIELDS:
        actual = _strict_int(delta[key], f"server_stats_delta.{key}")
        expected = (
            _strict_int(after[key], f"server_stats_after.{key}")
            if key == "current_queue_depth"
            else _strict_int(after[key], f"server_stats_after.{key}")
            - _strict_int(before[key], f"server_stats_before.{key}")
        )
        if actual != expected:
            raise ValueError(f"server_stats_delta.{key} is inconsistent with snapshots")
    return delta


def _weighted_mean_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    mean_key: str,
    count_key: str,
) -> float | None:
    count_before = int(before[count_key])
    count_after = int(after[count_key])
    count_delta = count_after - count_before
    if count_delta <= 0:
        return None
    return (
        float(after[mean_key]) * count_after - float(before[mean_key]) * count_before
    ) / count_delta


def _close_enough(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{context} is inconsistent with raw benchmark observations")


def _validate_samples(
    value: Any,
    case: Mapping[str, Any],
    *,
    image_count: int,
    context: str,
) -> list[dict[str, Any]]:
    requests = int(case["requests"])
    if not isinstance(value, list) or len(value) != requests:
        raise ValueError(f"{context}.samples must contain exactly {requests} entries")
    samples: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        sample = _object(raw, f"{context}.samples[{index}]")
        # Additive diagnostics keep historical schema-1 artifacts readable.
        optional_timings = {"server_processing_ms", "upload_and_parse_ms", "preprocessing_ms"}
        _exact_fields(
            sample, _SAMPLE_FIELDS | (sample.keys() & optional_timings),
            f"{context}.samples[{index}]",
        )
        for key in sample.keys() & optional_timings:
            _optional_finite(sample[key], f"{context}.samples[{index}].{key}")
        if _strict_int(sample["sequence"], f"{context}.samples[{index}].sequence") != index:
            raise ValueError(f"{context}.samples sequences must be contiguous and ordered")
        image_index = _strict_int(
            sample["image_index"],
            f"{context}.samples[{index}].image_index",
        )
        if image_index >= image_count:
            raise ValueError(f"{context}.samples[{index}].image_index is out of range")
        _finite_number(sample["elapsed_seconds"], f"{context}.samples[{index}].elapsed_seconds")
        status_code = sample["status_code"]
        if status_code is not None:
            _strict_int(status_code, f"{context}.samples[{index}].status_code", minimum=100)
        success = _strict_bool(sample["success"], f"{context}.samples[{index}].success")
        _strict_bool(sample["timed_out"], f"{context}.samples[{index}].timed_out")
        for key in ("error", "request_id"):
            if sample[key] is not None:
                _strict_text(sample[key], f"{context}.samples[{index}].{key}")
        for key in ("queue_wait_ms", "backend_inference_ms"):
            _optional_finite(sample[key], f"{context}.samples[{index}].{key}")
        realized = sample["realized_batch_size"]
        if realized is not None:
            _strict_int(realized, f"{context}.samples[{index}].realized_batch_size", minimum=1)
        if success:
            if not isinstance(status_code, int) or not 200 <= status_code < 300:
                raise ValueError(f"{context}.samples[{index}] success requires HTTP 2xx")
            if sample["error"] is not None or sample["request_id"] is None:
                raise ValueError(f"{context}.samples[{index}] success metadata is inconsistent")
            if any(
                sample[key] is None
                for key in ("queue_wait_ms", "backend_inference_ms", "realized_batch_size")
            ):
                raise ValueError(f"{context}.samples[{index}] success telemetry is incomplete")
        elif isinstance(status_code, int) and 200 <= status_code < 300:
            raise ValueError(f"{context}.samples[{index}] failure cannot have HTTP 2xx")
        samples.append(sample)
    return samples


def _validate_optional_metric(actual: Any, expected: float | None, context: str) -> None:
    value = _optional_finite(actual, context)
    if expected is None:
        if value is not None:
            raise ValueError(f"{context} must be null when no observations exist")
        return
    if value is None:
        raise ValueError(f"{context} is required when observations exist")
    _close_enough(value, expected, context)


def _validate_case(
    raw: Any,
    expected: Any,
    *,
    image_count: int,
    configured_device: str,
    index: int,
) -> dict[str, Any]:
    context = f"cases[{index}]"
    case = _object(raw, context)
    _exact_fields(case, _CASE_FIELDS, context)
    expected_values: dict[str, Any] = {
        "case_id": expected.case_id,
        "backend": expected.backend,
        "scheduler_mode": expected.scheduler_mode,
        "concurrency": expected.concurrency,
        "max_batch_size": expected.max_batch_size,
        "max_wait_ms": expected.max_wait_ms,
        "requests": expected.requests,
    }
    for key, expected_value in expected_values.items():
        actual = case[key]
        if isinstance(expected_value, int):
            actual = _strict_int(actual, f"{context}.{key}", minimum=1)
        elif isinstance(expected_value, float):
            actual = _finite_number(actual, f"{context}.{key}")
        else:
            actual = _strict_text(actual, f"{context}.{key}")
        if actual != expected_value:
            raise ValueError(f"{context}.{key} does not match the canonical matrix")
    device = _strict_text(case["device"], f"{context}.device")
    if device != configured_device:
        raise ValueError(f"{context}.device does not match benchmark_config")

    requests = int(expected.requests)
    successful = _strict_int(case["successful_requests"], f"{context}.successful_requests")
    failures = _strict_int(case["failures"], f"{context}.failures")
    rejections = _strict_int(case["rejections"], f"{context}.rejections")
    timeouts = _strict_int(case["timeouts"], f"{context}.timeouts")
    if successful + failures != requests:
        raise ValueError(f"{context} success/failure counts do not sum to requests")
    if rejections + timeouts > failures:
        raise ValueError(f"{context} rejection/timeout counts exceed failures")

    duration = _finite_number(case["duration_seconds"], f"{context}.duration_seconds")
    if duration <= 0:
        raise ValueError(f"{context}.duration_seconds must be positive")
    throughput = _finite_number(
        case["throughput_requests_per_second"],
        f"{context}.throughput_requests_per_second",
    )
    _close_enough(throughput, successful / duration, f"{context}.throughput_requests_per_second")
    backend_calls = _strict_int(
        case["backend_inference_call_count"],
        f"{context}.backend_inference_call_count",
    )
    batches = _strict_int(case["batches_executed"], f"{context}.batches_executed")
    if backend_calls != batches:
        raise ValueError(f"{context} backend call and batch counts must match")
    maximum_batch = _strict_int(
        case["maximum_realized_batch_size"],
        f"{context}.maximum_realized_batch_size",
    )
    if maximum_batch > int(expected.max_batch_size):
        raise ValueError(f"{context} realized batch exceeds configured maximum")

    samples = _validate_samples(
        case["samples"],
        case,
        image_count=image_count,
        context=context,
    )
    sample_successes = [sample for sample in samples if sample["success"] is True]
    if len(sample_successes) != successful:
        raise ValueError(f"{context}.successful_requests does not match raw samples")
    if sum(sample["success"] is False for sample in samples) != failures:
        raise ValueError(f"{context}.failures does not match raw samples")
    if sum(sample["status_code"] == 429 for sample in samples) != rejections:
        raise ValueError(f"{context}.rejections does not match raw samples")
    if sum(sample["timed_out"] is True for sample in samples) != timeouts:
        raise ValueError(f"{context}.timeouts does not match raw samples")
    request_ids = [sample["request_id"] for sample in sample_successes]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{context} successful request IDs must be unique")

    latency_ms = [float(sample["elapsed_seconds"]) * 1000.0 for sample in sample_successes]
    queue_waits = [float(sample["queue_wait_ms"]) for sample in sample_successes]
    realized_batches = [int(sample["realized_batch_size"]) for sample in sample_successes]
    if latency_ms:
        expected_latency = {
            "mean_latency_ms": float(np.mean(latency_ms)),
            "p50_latency_ms": float(np.percentile(latency_ms, 50.0)),
            "p95_latency_ms": float(np.percentile(latency_ms, 95.0)),
            "p99_latency_ms": float(np.percentile(latency_ms, 99.0)),
            "mean_queue_wait_ms": float(np.mean(queue_waits)),
            "p95_queue_wait_ms": float(np.percentile(queue_waits, 95.0)),
        }
        for key, expected_value in expected_latency.items():
            _validate_optional_metric(case[key], expected_value, f"{context}.{key}")
        if maximum_batch != max(realized_batches):
            raise ValueError(f"{context}.maximum_realized_batch_size is inconsistent")
    else:
        for key in (
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "mean_queue_wait_ms",
            "p95_queue_wait_ms",
        ):
            _validate_optional_metric(case[key], None, f"{context}.{key}")
        if maximum_batch != 0:
            raise ValueError(f"{context}.maximum_realized_batch_size must be zero")

    before = _validate_stats(case["server_stats_before"], f"{context}.server_stats_before")
    after = _validate_stats(case["server_stats_after"], f"{context}.server_stats_after")
    for key in _STATS_COUNTER_FIELDS - {"current_queue_depth"}:
        if int(after[key]) < int(before[key]):
            raise ValueError(f"{context}.server_stats_after.{key} cannot decrease")
    if float(after["uptime_seconds"]) < float(before["uptime_seconds"]):
        raise ValueError(f"{context} server uptime cannot decrease")
    delta = _validate_delta(case["server_stats_delta"], before, after)
    for snapshot in (before, after):
        if snapshot["backend"] != case["backend"] or snapshot["device"] != device:
            raise ValueError(f"{context} backend/device provenance is inconsistent")
        if snapshot["scheduler_mode"] != case["scheduler_mode"]:
            raise ValueError(f"{context} scheduler provenance is inconsistent")
    if int(delta["completed_requests"]) != successful:
        raise ValueError(f"{context} completed request delta is inconsistent")
    if int(delta["rejected_requests"]) != rejections:
        raise ValueError(f"{context} rejected request delta is inconsistent")
    if int(delta["timed_out_requests"]) != timeouts:
        raise ValueError(f"{context} timeout delta is inconsistent")
    accounted_failures = sum(
        int(delta[key])
        for key in (
            "rejected_requests",
            "timed_out_requests",
            "cancelled_requests",
            "failed_requests",
        )
    )
    if accounted_failures != failures:
        raise ValueError(f"{context} server failure deltas are inconsistent")
    if int(delta["total_accepted_requests"]) != requests - rejections:
        raise ValueError(f"{context} accepted request delta is inconsistent")
    if int(delta["backend_inference_calls"]) != backend_calls:
        raise ValueError(f"{context} backend call delta is inconsistent")
    if int(delta["batches_executed"]) != batches:
        raise ValueError(f"{context} batch delta is inconsistent")
    if int(delta["current_queue_depth"]) != 0:
        raise ValueError(f"{context} queue must be drained after measurement")

    backend_mean = _weighted_mean_delta(
        before,
        after,
        mean_key="mean_backend_inference_ms",
        count_key="backend_inference_calls",
    )
    batch_mean = _weighted_mean_delta(
        before,
        after,
        mean_key="mean_realized_batch_size",
        count_key="batches_executed",
    )
    _validate_optional_metric(
        case["mean_backend_inference_ms"],
        backend_mean,
        f"{context}.mean_backend_inference_ms",
    )
    _validate_optional_metric(
        case["mean_realized_batch_size"],
        batch_mean,
        f"{context}.mean_realized_batch_size",
    )
    if batch_mean is not None and not 1.0 <= batch_mean <= float(expected.max_batch_size):
        raise ValueError(f"{context}.mean_realized_batch_size is outside configured bounds")
    return case


def _validate_cases(value: Any, settings: Any, image_count: int) -> list[dict[str, Any]]:
    from inference_gateway.benchmark import build_primary_cases

    expected_cases = build_primary_cases(settings)
    if not isinstance(value, list) or len(value) != len(expected_cases):
        raise ValueError(f"cases must contain exactly {len(expected_cases)} primary cases")
    actual_ids = [
        _strict_text(_object(case, f"cases[{index}]").get("case_id"), f"cases[{index}].case_id")
        for index, case in enumerate(value)
    ]
    expected_ids = [case.case_id for case in expected_cases]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("case IDs must be unique")
    if actual_ids != expected_ids:
        raise ValueError("case IDs and order must match the canonical 32-case matrix")
    return [
        _validate_case(
            raw,
            expected,
            image_count=image_count,
            configured_device=settings.model.device.value,
            index=index,
        )
        for index, (raw, expected) in enumerate(zip(value, expected_cases, strict=True))
    ]


def _validate_generated_at(value: Any) -> str:
    text = _strict_text(value, "generated_at_utc")
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("generated_at_utc must use a UTC offset")
    return text


def _validate_results_body(results: dict[str, Any]) -> None:
    version = _strict_int(results.get("schema_version"), "schema_version", minimum=1)
    if version != BENCHMARK_RESULTS_SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark results schema version {version}")
    _validate_generated_at(results.get("generated_at_utc"))
    _validate_environment(results.get("environment"))
    settings = _validate_config(results.get("benchmark_config"))
    _validate_parity(results.get("parity_artifact"), settings)
    images = _validate_images(
        results.get("image_sequence"),
        settings.benchmark.synthetic_image_count,
    )
    _validate_cases(results.get("cases"), settings, len(images))


def _validate_provenance(results: Mapping[str, Any]) -> None:
    provenance = _object(results.get("provenance"), "provenance")
    _exact_fields(provenance, _PROVENANCE_FIELDS, "provenance")
    if provenance["algorithm"] != _PROVENANCE_ALGORITHM:
        raise ValueError("unsupported benchmark provenance algorithm")
    base_payload = {key: results[key] for key in _RESULT_FIELDS}
    expected = {
        "payload_sha256": _sha256_digest(base_payload),
        "benchmark_config_sha256": _sha256_digest(results["benchmark_config"]),
        "parity_artifact_sha256": _sha256_digest(results["parity_artifact"]),
    }
    for key, expected_digest in expected.items():
        actual = _strict_text(provenance[key], f"provenance.{key}")
        if actual != expected_digest:
            raise ValueError(f"provenance.{key} does not match benchmark payload")


def validate_benchmark_results(
    results: Mapping[str, Any],
    *,
    require_provenance: bool = True,
) -> dict[str, Any]:
    """Strictly validate the canonical benchmark schema before making claims."""

    payload = _object(results, "benchmark results")
    expected_fields = _RESULT_FIELDS | ({"provenance"} if require_provenance else set())
    _exact_fields(payload, frozenset(expected_fields), "benchmark results")
    _validate_results_body(payload)
    if require_provenance:
        _validate_provenance(payload)
    return payload


def seal_benchmark_results(results: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a freshly measured canonical payload with reproducible provenance hashes."""

    unsealed = {key: value for key, value in results.items() if key != "provenance"}
    validate_benchmark_results(unsealed, require_provenance=False)
    try:
        payload: dict[str, Any] = json.loads(
            json.dumps(unsealed, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("benchmark results must contain only finite JSON values") from error
    payload["provenance"] = {
        "algorithm": _PROVENANCE_ALGORITHM,
        "payload_sha256": _sha256_digest(payload),
        "benchmark_config_sha256": _sha256_digest(payload["benchmark_config"]),
        "parity_artifact_sha256": _sha256_digest(payload["parity_artifact"]),
    }
    validate_benchmark_results(payload)
    return payload


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a benchmark JSON object for report regeneration."""

    results_path = Path(path)
    with results_path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"expected a JSON object in {results_path}")
    return validate_benchmark_results(payload)


def _cases(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = results.get("cases")
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("benchmark results must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(candidate):
        if not isinstance(case, dict) or not all(isinstance(key, str) for key in case):
            raise ValueError(f"benchmark case {index} is not a JSON object")
        cases.append(case)
    return cases


def _number(case: Mapping[str, Any], key: str) -> float | None:
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(case: Mapping[str, Any], key: str) -> int:
    value = case.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _text(case: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = case.get(key)
    return value if isinstance(value, str) else default


def _format_number(value: Any, *, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _best_case(
    cases: Sequence[Mapping[str, Any]],
    key: str,
    *,
    maximize: bool,
) -> Mapping[str, Any] | None:
    candidates = [
        case
        for case in cases
        if _fully_successful(case) and _number(case, key) is not None
    ]
    if not candidates:
        return None
    return (max if maximize else min)(
        candidates,
        key=lambda case: float(_number(case, key) or 0.0),
    )


def _fully_successful(case: Mapping[str, Any]) -> bool:
    requested = _integer(case, "requests")
    return (
        requested > 0
        and _integer(case, "successful_requests") == requested
        and _integer(case, "failures") == 0
        and _integer(case, "rejections") == 0
        and _integer(case, "timeouts") == 0
    )


def _case_label(case: Mapping[str, Any]) -> str:
    return _text(case, "case_id", "unnamed case")


def _json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _csv_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = {field: case.get(field) for field in _CSV_FIELDS}
        delta = case.get("server_stats_delta")
        if isinstance(delta, dict):
            for key, value in delta.items():
                if isinstance(key, str) and not isinstance(value, (dict, list)):
                    row[f"stats_delta_{key}"] = value
        rows.append(row)
    return rows


def _write_csv(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(_csv_rows(cases))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _group_plot_points(
    cases: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> dict[str, list[tuple[float, float, str]]]:
    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for case in cases:
        x_value = _number(case, x_key)
        y_value = _number(case, y_key)
        if x_value is None or y_value is None:
            continue
        label = (
            f"{_text(case, 'backend')} / {_text(case, 'scheduler_mode')} / "
            f"{_text(case, 'device')}"
        )
        groups[label].append((x_value, y_value, _case_label(case)))
    return groups


def _save_scatter(
    groups: Mapping[str, Sequence[tuple[float, float, str]]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    if groups:
        for index, (label, points) in enumerate(sorted(groups.items())):
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                label=label,
                marker=markers[index % len(markers)],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.6,
                s=58,
            )
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No successful benchmark measurements",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": "adaptive-inference-gateway"})
    plt.close(fig)


def _write_charts(cases: Sequence[Mapping[str, Any]], output_directory: Path) -> tuple[Path, Path]:
    throughput_path = output_directory / "throughput_vs_p95.png"
    batch_path = output_directory / "batch_efficiency.png"
    _save_scatter(
        _group_plot_points(
            cases,
            x_key="p95_latency_ms",
            y_key="throughput_requests_per_second",
        ),
        x_label="p95 latency (ms)",
        y_label="Throughput (successful requests/s)",
        title="Throughput versus tail latency",
        path=throughput_path,
    )
    _save_scatter(
        _group_plot_points(
            cases,
            x_key="mean_realized_batch_size",
            y_key="throughput_requests_per_second",
        ),
        x_label="Mean realized batch size",
        y_label="Throughput (successful requests/s)",
        title="Realized batching efficiency",
        path=batch_path,
    )
    return throughput_path, batch_path


def _environment_lines(environment: Mapping[str, Any]) -> list[str]:
    fields = (
        ("Operating system", "operating_system"),
        ("Machine", "machine"),
        ("Processor", "processor"),
        ("Logical CPUs", "logical_cpu_count"),
        ("Python", "python_version"),
        ("PyTorch", "torch_version"),
        ("torchvision", "torchvision_version"),
        ("ONNX", "onnx_version"),
        ("ONNX Runtime", "onnxruntime_version"),
        ("CUDA available", "cuda_available"),
        ("CUDA version", "cuda_version"),
        ("CUDA device", "cuda_device_name"),
    )
    return [f"- {label}: `{environment.get(key, 'unknown')}`" for label, key in fields]


def _results_table(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Case | Device | C | Batch | Wait ms | Success | Failures | Reject | Timeout | "
        "req/s | Mean ms | p50 ms | p95 ms | p99 ms | Mean queue ms | Mean backend ms | "
        "Mean realized batch | Max realized batch | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                (
                    _case_label(case),
                    _text(case, "device"),
                    str(_integer(case, "concurrency")),
                    str(_integer(case, "max_batch_size")),
                    _format_number(case.get("max_wait_ms"), digits=1),
                    str(_integer(case, "successful_requests")),
                    str(_integer(case, "failures")),
                    str(_integer(case, "rejections")),
                    str(_integer(case, "timeouts")),
                    _format_number(case.get("throughput_requests_per_second")),
                    _format_number(case.get("mean_latency_ms")),
                    _format_number(case.get("p50_latency_ms")),
                    _format_number(case.get("p95_latency_ms")),
                    _format_number(case.get("p99_latency_ms")),
                    _format_number(case.get("mean_queue_wait_ms")),
                    _format_number(case.get("mean_backend_inference_ms")),
                    _format_number(case.get("mean_realized_batch_size")),
                    str(_integer(case, "maximum_realized_batch_size")),
                    str(_integer(case, "backend_inference_call_count")),
                )
            )
            + " |"
        )
    return lines


def _batching_tradeoff_lines(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for backend in ("torch", "onnx"):
        devices = sorted(
            {
                _text(case, "device")
                for case in cases
                if _text(case, "backend") == backend
                and _text(case, "device") != "unknown"
            }
        )
        for device in devices:
            for concurrency in sorted({_integer(case, "concurrency") for case in cases}):
                direct = [
                    case
                    for case in cases
                    if _text(case, "backend") == backend
                    and _text(case, "device") == device
                    and _text(case, "scheduler_mode") == "direct"
                    and _integer(case, "concurrency") == concurrency
                ]
                dynamic = [
                    case
                    for case in cases
                    if _text(case, "backend") == backend
                    and _text(case, "device") == device
                    and _text(case, "scheduler_mode") == "dynamic"
                    and _integer(case, "concurrency") == concurrency
                ]
                direct_case = _best_case(
                    direct,
                    "throughput_requests_per_second",
                    maximize=True,
                )
                best_dynamic = _best_case(
                    dynamic,
                    "throughput_requests_per_second",
                    maximize=True,
                )
                if direct_case is None or best_dynamic is None:
                    continue
                direct_throughput = _number(direct_case, "throughput_requests_per_second")
                dynamic_throughput = _number(
                    best_dynamic,
                    "throughput_requests_per_second",
                )
                direct_p95 = _number(direct_case, "p95_latency_ms")
                dynamic_p95 = _number(best_dynamic, "p95_latency_ms")
                if (
                    direct_throughput is None
                    or dynamic_throughput is None
                    or direct_p95 is None
                    or dynamic_p95 is None
                ):
                    continue
                throughput_change = (
                    100.0 * (dynamic_throughput - direct_throughput) / direct_throughput
                    if direct_throughput != 0.0
                    else 0.0
                )
                p95_change = (
                    100.0 * (dynamic_p95 - direct_p95) / direct_p95
                    if direct_p95 != 0.0
                    else 0.0
                )
                lines.append(
                    f"- {backend} on {device} at concurrency {concurrency}: "
                    f"best-throughput dynamic case `{_case_label(best_dynamic)}` changed "
                    f"throughput by {throughput_change:+.2f}% and p95 latency by "
                    f"{p95_change:+.2f}% relative to direct."
                )
    if not lines:
        lines.append(
            "- No matched fully successful direct/dynamic measurements were available."
        )
    return lines


def _backend_comparison_lines(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    direct_cases = [case for case in cases if _text(case, "scheduler_mode") == "direct"]
    devices = sorted(
        {
            _text(case, "device")
            for case in direct_cases
            if _text(case, "device") != "unknown"
        }
    )
    for device in devices:
        for concurrency in sorted({_integer(case, "concurrency") for case in direct_cases}):
            torch_case = _best_case(
                [
                    case
                    for case in direct_cases
                    if _text(case, "backend") == "torch"
                    and _text(case, "device") == device
                    and _integer(case, "concurrency") == concurrency
                ],
                "throughput_requests_per_second",
                maximize=True,
            )
            onnx_case = _best_case(
                [
                    case
                    for case in direct_cases
                    if _text(case, "backend") == "onnx"
                    and _text(case, "device") == device
                    and _integer(case, "concurrency") == concurrency
                ],
                "throughput_requests_per_second",
                maximize=True,
            )
            if torch_case is None or onnx_case is None:
                continue
            torch_throughput = _number(torch_case, "throughput_requests_per_second")
            onnx_throughput = _number(onnx_case, "throughput_requests_per_second")
            torch_p95 = _number(torch_case, "p95_latency_ms")
            onnx_p95 = _number(onnx_case, "p95_latency_ms")
            if (
                torch_throughput is None
                or onnx_throughput is None
                or torch_p95 is None
                or onnx_p95 is None
            ):
                continue
            lines.append(
                f"- Direct {device} concurrency {concurrency}: PyTorch "
                f"{torch_throughput:.3f} req/s at {torch_p95:.3f} ms p95; "
                f"ONNX {onnx_throughput:.3f} req/s at {onnx_p95:.3f} ms p95."
            )
    if not lines:
        lines.append(
            "- No matched fully successful PyTorch/ONNX direct measurements were available."
        )
    return lines


def render_report(results: Mapping[str, Any]) -> str:
    """Render a Markdown report solely from measured result data."""

    validate_benchmark_results(results)
    cases = _cases(results)
    environment_candidate = results.get("environment", {})
    environment = (
        environment_candidate if isinstance(environment_candidate, dict) else {}
    )
    config_candidate = results.get("benchmark_config", {})
    config = config_candidate if isinstance(config_candidate, dict) else {}
    parity_candidate = results.get("parity_artifact", {})
    parity = parity_candidate if isinstance(parity_candidate, dict) else {}

    best_throughput = _best_case(
        cases,
        "throughput_requests_per_second",
        maximize=True,
    )
    best_p95 = _best_case(cases, "p95_latency_ms", maximize=False)
    best_lines = []
    if best_throughput is not None:
        best_lines.append(
            f"- Best measured throughput: `{_case_label(best_throughput)}` at "
            f"{_format_number(best_throughput.get('throughput_requests_per_second'))} "
            f"successful requests/s on `{_text(best_throughput, 'device')}`."
        )
    if best_p95 is not None:
        best_lines.append(
            f"- Best measured p95 latency: `{_case_label(best_p95)}` at "
            f"{_format_number(best_p95.get('p95_latency_ms'))} ms on "
            f"`{_text(best_p95, 'device')}`."
        )
    if not best_lines:
        best_lines.append("- No fully successful measurements were available.")

    benchmark_config = config.get("benchmark", {})
    scheduler_config = config.get("scheduler", {})
    requests_per_case = (
        benchmark_config.get("requests_per_case", "unknown")
        if isinstance(benchmark_config, dict)
        else "unknown"
    )
    warmup_requests = (
        benchmark_config.get("warmup_requests", "unknown")
        if isinstance(benchmark_config, dict)
        else "unknown"
    )
    request_timeout_ms = (
        scheduler_config.get("request_timeout_ms", "unknown")
        if isinstance(scheduler_config, dict)
        else "unknown"
    )
    lines = [
        "# Adaptive Inference Gateway Benchmark Report",
        "",
        f"Generated: `{results.get('generated_at_utc', 'unknown')}`",
        "",
        "## Environment",
        "",
        *_environment_lines(environment),
        "",
        "## Experiment configuration",
        "",
        f"- Completed primary cases: `{len(cases)}`",
        f"- Requests per case: `{requests_per_case}`",
        f"- Warmup requests per case: `{warmup_requests}`",
        f"- Request timeout ms: `{request_timeout_ms}`",
        "- Inputs: the fixed synthetic image sequence recorded by SHA-256 in `results.json`.",
        "- Latency percentiles: calculated from individual successful HTTP request "
        "samples; warmups are excluded.",
        "- Throughput: successful measured responses divided by measured wall-clock duration.",
        "",
        "## ONNX parity gate",
        "",
        f"- Passed: `{parity.get('passed', 'unknown')}`",
        f"- ONNX SHA-256: `{parity.get('onnx_sha256', 'unknown')}`",
        f"- Maximum absolute logit difference: `{parity.get('max_abs_difference', 'unknown')}`",
        f"- Mean absolute logit difference: `{parity.get('mean_abs_difference', 'unknown')}`",
        f"- Top-1 agreement: `{parity.get('top1_agreement', 'unknown')}`",
        "",
        "## Highlights",
        "",
        *best_lines,
        "",
        "## Full primary results",
        "",
        *_results_table(cases),
        "",
        "## Observed batching trade-offs",
        "",
        *_batching_tradeoff_lines(cases),
        "",
        "Positive changes mean an increase; a throughput increase and a p95 increase "
        "therefore describe a throughput/latency trade-off, not an unqualified improvement.",
        "",
        "## PyTorch versus ONNX",
        "",
        *_backend_comparison_lines(cases),
        "",
        "No backend is assumed faster; the statements above are generated from this run.",
        "",
        "## Charts",
        "",
        "![Throughput versus p95 latency](throughput_vs_p95.png)",
        "",
        "![Realized batch size versus throughput](batch_efficiency.png)",
        "",
        "## Limitations",
        "",
        "- The synthetic inputs make serving runs reproducible but do not measure "
        "ImageNet accuracy.",
        "- These results describe only the recorded hardware, software versions, "
        "and configuration.",
        "- Client and server share one host, so contention and loopback transport "
        "affect measurements.",
        "- Each configuration was measured once in a fixed case order; run-to-run "
        "variance and order effects were not quantified.",
        "- Thermal throttling, background processes, and changing cache state may "
        "influence throughput and latency.",
        "- This educational scheduler demonstrates production concepts; it is not "
        "a replacement for NVIDIA Triton.",
        "- Results from differently labeled CPU and CUDA runs must not be combined "
        "as if they were one environment.",
        "",
    ]
    return "\n".join(lines)


def _readme_excerpt(results: Mapping[str, Any], output_directory: Path, readme_path: Path) -> str:
    cases = _cases(results)
    environment_candidate = results.get("environment", {})
    environment = (
        environment_candidate if isinstance(environment_candidate, dict) else {}
    )
    best_throughput = _best_case(
        cases,
        "throughput_requests_per_second",
        maximize=True,
    )
    best_p95 = _best_case(cases, "p95_latency_ms", maximize=False)
    throughput_chart = os.path.relpath(
        output_directory / "throughput_vs_p95.png",
        readme_path.parent,
    ).replace("\\", "/")
    batch_chart = os.path.relpath(
        output_directory / "batch_efficiency.png",
        readme_path.parent,
    ).replace("\\", "/")
    report_path = os.path.relpath(
        output_directory / "report.md",
        readme_path.parent,
    ).replace("\\", "/")
    lines = [
        README_RESULTS_START,
        "## Measured benchmark results",
        "",
        f"This benchmark completed {len(cases)} primary cases on the environment "
        "recorded in `artifacts/results.json`.",
        "",
        "Measured environment:",
        "",
        f"- Operating system: `{environment.get('operating_system', 'unknown')}`",
        f"- Machine: `{environment.get('machine', 'unknown')}`",
        f"- Processor: `{environment.get('processor', 'unknown')}`",
        f"- Logical CPUs: `{environment.get('logical_cpu_count', 'unknown')}`",
        f"- Python: `{environment.get('python_version', 'unknown')}`",
    ]
    if best_throughput is not None:
        lines.extend(
            (
                "",
                f"- Best throughput: `{_case_label(best_throughput)}` — "
                f"{_format_number(best_throughput.get('throughput_requests_per_second'))} "
                f"req/s on `{_text(best_throughput, 'device')}`",
            )
        )
    if best_p95 is not None:
        lines.extend(
            (
                f"- Best p95 latency: `{_case_label(best_p95)}` — "
                f"{_format_number(best_p95.get('p95_latency_ms'))} ms on "
                f"`{_text(best_p95, 'device')}`",
                "",
            )
        )
    lines.extend(
        (
            f"![Throughput versus p95 latency]({throughput_chart})",
            "",
            f"![Realized batch size versus throughput]({batch_chart})",
            "",
            f"See [`{report_path}`]({report_path}) for the full generated table and "
            "methodology.",
            README_RESULTS_END,
        )
    )
    return "\n".join(lines)


def update_marked_readme(
    readme_path: str | Path,
    results: Mapping[str, Any],
    output_directory: str | Path,
) -> Path:
    """Replace an explicitly marked README section with measured results."""

    validate_benchmark_results(results)
    path = Path(readme_path)
    source = path.read_text(encoding="utf-8")
    start = source.find(README_RESULTS_START)
    end = source.find(README_RESULTS_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            f"{path} must contain ordered {README_RESULTS_START} and {README_RESULTS_END} markers"
        )
    end += len(README_RESULTS_END)
    replacement = _readme_excerpt(results, Path(output_directory), path)
    updated = source[:start] + replacement + source[end:]
    path.write_text(updated, encoding="utf-8", newline="\n")
    return path


def write_benchmark_artifacts(
    results: Mapping[str, Any],
    output_directory: str | Path,
    *,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> ArtifactPaths:
    """Write JSON, CSV, Markdown, and both charts from one results object."""

    validate_benchmark_results(results)
    cases = _cases(results)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    results_json = output_path / "results.json"
    results_csv = output_path / "results.csv"
    report_path = output_path / "report.md"

    _json_dump(results, results_json)
    _write_csv(cases, results_csv)
    throughput_chart, batch_chart = _write_charts(cases, output_path)
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    if update_readme:
        update_marked_readme(readme_path, results, output_path)
    return ArtifactPaths(
        results_json=results_json,
        results_csv=results_csv,
        report_markdown=report_path,
        throughput_vs_p95_chart=throughput_chart,
        batch_efficiency_chart=batch_chart,
    )


def generate_report(
    results_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> ArtifactPaths:
    """Regenerate all derived artifacts from an existing results.json."""

    source_path = Path(results_path)
    results = load_results(source_path)
    destination = Path(output_directory) if output_directory is not None else source_path.parent
    return write_benchmark_artifacts(
        results,
        destination,
        update_readme=update_readme,
        readme_path=readme_path,
    )


__all__ = [
    "BENCHMARK_RESULTS_SCHEMA_VERSION",
    "README_RESULTS_END",
    "README_RESULTS_START",
    "ArtifactPaths",
    "generate_report",
    "load_results",
    "render_report",
    "seal_benchmark_results",
    "update_marked_readme",
    "validate_benchmark_results",
    "write_benchmark_artifacts",
]

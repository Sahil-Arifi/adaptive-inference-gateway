from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

import inference_gateway.benchmark as benchmark_module
import inference_gateway.reporting as reporting_module
from inference_gateway.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    build_primary_cases,
    group_primary_cases,
    run_benchmark,
    run_benchmark_sync,
)
from inference_gateway.config import GatewaySettings
from inference_gateway.loadgen import LoadTestResult, RequestSample, generate_synthetic_images
from inference_gateway.reporting import (
    README_RESULTS_END,
    README_RESULTS_START,
    generate_report,
    load_results,
    render_report,
    seal_benchmark_results,
    update_marked_readme,
    validate_benchmark_results,
    write_benchmark_artifacts,
)


def _settings(tmp_path: Path | None = None) -> GatewaySettings:
    payload: dict[str, Any] = {
        "model": {"device": "cpu"},
        "benchmark": {"requests_per_case": 8},
    }
    if tmp_path is not None:
        payload["output"] = {"directory": str(tmp_path / "artifacts")}
    return GatewaySettings.model_validate(payload)


def _stats_snapshot(
    case: BenchmarkCase,
    *,
    requests: int,
    successful: int,
    failures: int,
    calls: int,
    batch_size: int,
    before: bool,
) -> dict[str, Any]:
    if before:
        requests = successful = failures = calls = 0
        batch_size = 0
    return {
        "total_accepted_requests": requests,
        "completed_requests": successful,
        "rejected_requests": 0,
        "timed_out_requests": 0,
        "cancelled_requests": 0,
        "failed_requests": failures,
        "backend_inference_calls": calls,
        "batches_executed": calls,
        "current_queue_depth": 0,
        "maximum_observed_queue_depth": 0 if before else case.concurrency,
        "mean_realized_batch_size": 0.0 if before else float(batch_size),
        "maximum_realized_batch_size": batch_size,
        "mean_queue_wait_ms": 0.0 if before else 1.25,
        "mean_backend_inference_ms": 0.0 if before else 4.5,
        "scheduler_mode": case.scheduler_mode,
        "backend": case.backend,
        "device": "cpu",
        "uptime_seconds": 1.0 if before else 2.0,
    }


def _case_result(
    case: BenchmarkCase,
    *,
    index: int = 0,
    throughput: float | None = None,
    p95_latency_ms: float | None = None,
) -> BenchmarkCaseResult:
    measured_throughput = throughput if throughput is not None else 100.0 + index
    measured_p95 = p95_latency_ms if p95_latency_ms is not None else 20.0 + index
    batch_size = 1 if case.scheduler_mode == "direct" else 4
    calls = case.requests if case.scheduler_mode == "direct" else case.requests // batch_size
    samples = tuple(
        RequestSample(
            sequence=sequence,
            image_index=sequence % 8,
            elapsed_seconds=measured_p95 / 1000.0,
            status_code=200,
            success=True,
            timed_out=False,
            error=None,
            request_id=f"request-{index}-{sequence}",
            queue_wait_ms=1.25,
            backend_inference_ms=4.5,
            realized_batch_size=batch_size,
        )
        for sequence in range(case.requests)
    )
    before = _stats_snapshot(
        case,
        requests=case.requests,
        successful=case.requests,
        failures=0,
        calls=calls,
        batch_size=batch_size,
        before=True,
    )
    after = _stats_snapshot(
        case,
        requests=case.requests,
        successful=case.requests,
        failures=0,
        calls=calls,
        batch_size=batch_size,
        before=False,
    )
    return BenchmarkCaseResult(
        case_id=case.case_id,
        backend=case.backend,
        device="cpu",
        scheduler_mode=case.scheduler_mode,
        concurrency=case.concurrency,
        max_batch_size=case.max_batch_size,
        max_wait_ms=case.max_wait_ms,
        requests=case.requests,
        successful_requests=case.requests,
        failures=0,
        rejections=0,
        timeouts=0,
        throughput_requests_per_second=measured_throughput,
        mean_latency_ms=measured_p95,
        p50_latency_ms=measured_p95,
        p95_latency_ms=measured_p95,
        p99_latency_ms=measured_p95,
        mean_queue_wait_ms=1.25,
        p95_queue_wait_ms=1.25,
        mean_backend_inference_ms=4.5,
        mean_realized_batch_size=float(batch_size),
        maximum_realized_batch_size=batch_size,
        backend_inference_call_count=calls,
        batches_executed=calls,
        duration_seconds=case.requests / measured_throughput,
        server_stats_delta={
            "total_accepted_requests": case.requests,
            "completed_requests": case.requests,
            "rejected_requests": 0,
            "timed_out_requests": 0,
            "cancelled_requests": 0,
            "failed_requests": 0,
            "backend_inference_calls": calls,
            "batches_executed": calls,
            "current_queue_depth": 0,
        },
        server_stats_before=before,
        server_stats_after=after,
        samples=samples,
    )


def _synthetic_results(tmp_path: Path | None = None) -> dict[str, Any]:
    settings = _settings(tmp_path)
    primary_cases = build_primary_cases(settings)
    case_results = [
        _case_result(case, index=index)
        for index, case in enumerate(primary_cases)
    ]
    # Make the extrema unambiguous and retain a long binary float for round-trip checks.
    case_results[0] = _case_result(
        primary_cases[0],
        index=0,
        throughput=123.12345678901234,
        p95_latency_ms=1.2345678901234567,
    )
    case_results[-1] = _case_result(
        primary_cases[-1],
        index=len(primary_cases) - 1,
        throughput=999.9876543210987,
        p95_latency_ms=80.0,
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-08-27T12:34:56+00:00",
        "environment": {
            "operating_system": "TestOS 1.0",
            "system": "TestOS",
            "release": "1.0",
            "machine": "test-machine",
            "processor": "test-cpu",
            "logical_cpu_count": 8,
            "python_version": "3.11.9",
            "torch_version": "test",
            "torchvision_version": "test",
            "onnx_version": "test",
            "onnxruntime_version": "test",
            "onnxruntime_available_providers": ["CPUExecutionProvider"],
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_name": None,
        },
        "benchmark_config": settings.model_dump(mode="json"),
        "parity_artifact": {
            "schema_version": 1,
            "model_name": "resnet18",
            "onnx_path": str(settings.model.onnx_path),
            "passed": True,
            "onnx_sha256": "a" * 64,
            "input_name": "images",
            "output_name": "logits",
            "input_shape": [3, 224, 224],
            "batch_sizes": [1, 4, 16],
            "rtol": 1e-4,
            "atol": 1e-5,
            "seed": 2027,
            "device": "cpu",
            "per_batch": [
                {
                    "batch_size": batch_size,
                    "logit_count": batch_size * 1000,
                    "max_abs_difference": 1e-6,
                    "mean_abs_difference": 1e-7,
                    "top1_agreement": 1.0,
                    "allclose": True,
                    "passed": True,
                }
                for batch_size in (1, 4, 16)
            ],
            "max_abs_difference": 1e-6,
            "mean_abs_difference": 1e-7,
            "top1_agreement": 1.0,
        },
        "image_sequence": [
            {
                "sequence_index": index,
                "filename": f"synthetic-{index:03d}.{'png' if index % 2 == 0 else 'jpg'}",
                "media_type": "image/png" if index % 2 == 0 else "image/jpeg",
                "byte_length": 100 + index,
                "sha256": format(index, "x") * 64,
            }
            for index in range(settings.benchmark.synthetic_image_count)
        ],
        "cases": [result.to_dict() for result in case_results],
    }
    return seal_benchmark_results(payload)


def _set_case_success_count(
    case: dict[str, Any],
    successful: int,
    *,
    throughput: float | None = None,
    latency_ms: float | None = None,
) -> None:
    requests = int(case["requests"])
    failures = requests - successful
    batch_size = 1 if case["scheduler_mode"] == "direct" else 4
    calls = 0 if successful == 0 else (successful + batch_size - 1) // batch_size
    measured_latency = (
        float(case["p95_latency_ms"]) if latency_ms is None else latency_ms
    )
    measured_throughput = (
        float(case["throughput_requests_per_second"])
        if throughput is None
        else throughput
    )
    for sequence, sample in enumerate(case["samples"]):
        if sequence < successful:
            sample.update(
                {
                    "elapsed_seconds": measured_latency / 1000.0,
                    "status_code": 200,
                    "success": True,
                    "timed_out": False,
                    "error": None,
                    "request_id": f"partial-{case['case_id']}-{sequence}",
                    "queue_wait_ms": 1.25,
                    "backend_inference_ms": 4.5,
                    "realized_batch_size": batch_size,
                }
            )
        else:
            sample.update(
                {
                    "status_code": 500,
                    "success": False,
                    "timed_out": False,
                    "error": "synthetic failure",
                    "request_id": None,
                    "queue_wait_ms": None,
                    "backend_inference_ms": None,
                    "realized_batch_size": None,
                }
            )

    case.update(
        {
            "successful_requests": successful,
            "failures": failures,
            "rejections": 0,
            "timeouts": 0,
            "throughput_requests_per_second": (
                measured_throughput if successful else 0.0
            ),
            "mean_latency_ms": measured_latency if successful else None,
            "p50_latency_ms": measured_latency if successful else None,
            "p95_latency_ms": measured_latency if successful else None,
            "p99_latency_ms": measured_latency if successful else None,
            "mean_queue_wait_ms": 1.25 if successful else None,
            "p95_queue_wait_ms": 1.25 if successful else None,
            "mean_backend_inference_ms": 4.5 if calls else None,
            "mean_realized_batch_size": successful / calls if calls else None,
            "maximum_realized_batch_size": batch_size if successful else 0,
            "backend_inference_call_count": calls,
            "batches_executed": calls,
            "duration_seconds": (
                successful / measured_throughput
                if successful
                else float(case["duration_seconds"])
            ),
        }
    )
    before = _stats_snapshot(
        BenchmarkCase(
            case_id=case["case_id"],
            backend=case["backend"],
            scheduler_mode=case["scheduler_mode"],
            concurrency=case["concurrency"],
            max_batch_size=case["max_batch_size"],
            max_wait_ms=case["max_wait_ms"],
            requests=requests,
        ),
        requests=requests,
        successful=successful,
        failures=failures,
        calls=calls,
        batch_size=batch_size,
        before=True,
    )
    after = dict(before)
    after.update(
        {
            "total_accepted_requests": requests,
            "completed_requests": successful,
            "failed_requests": failures,
            "backend_inference_calls": calls,
            "batches_executed": calls,
            "maximum_observed_queue_depth": case["concurrency"],
            "mean_realized_batch_size": successful / calls if calls else 0.0,
            "maximum_realized_batch_size": batch_size if successful else 0,
            "mean_queue_wait_ms": 1.25 if successful else 0.0,
            "mean_backend_inference_ms": 4.5 if calls else 0.0,
            "uptime_seconds": 2.0,
        }
    )
    case["server_stats_before"] = before
    case["server_stats_after"] = after
    case["server_stats_delta"] = {
        "total_accepted_requests": requests,
        "completed_requests": successful,
        "rejected_requests": 0,
        "timed_out_requests": 0,
        "cancelled_requests": 0,
        "failed_requests": failures,
        "backend_inference_calls": calls,
        "batches_executed": calls,
        "current_queue_depth": 0,
    }


def _reseal(results: Mapping[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(results)
    mutated.pop("provenance", None)
    return seal_benchmark_results(mutated)


def test_primary_matrix_has_exact_required_cases_and_ten_server_groups() -> None:
    cases = build_primary_cases(_settings())
    groups = group_primary_cases(cases)

    assert len(cases) == 32
    assert len({case.case_id for case in cases}) == 32
    assert sum(case.scheduler_mode == "direct" for case in cases) == 8
    assert sum(case.scheduler_mode == "dynamic" for case in cases) == 24
    assert {case.concurrency for case in cases if case.scheduler_mode == "direct"} == {
        1,
        8,
        32,
        64,
    }
    assert {case.concurrency for case in cases if case.scheduler_mode == "dynamic"} == {
        8,
        32,
        64,
    }
    assert len(groups) == 10
    for group in groups:
        assert len({case.server_group for case in group}) == 1


def test_matrix_and_grouping_fail_closed_when_invariants_are_bypassed() -> None:
    settings = _settings()
    invalid_benchmark = settings.benchmark.model_copy(update={"concurrency": [1, 8]})
    invalid_settings = settings.model_copy(update={"benchmark": invalid_benchmark})

    with pytest.raises(ValueError, match="exactly 32 unique"):
        build_primary_cases(invalid_settings)

    with pytest.raises(ValueError, match="expected 10"):
        group_primary_cases(build_primary_cases(settings)[:1])


def test_case_result_uses_per_case_stats_deltas_and_raw_request_telemetry() -> None:
    case = build_primary_cases(_settings())[0]
    samples = (
        RequestSample(0, 0, 0.01, 200, True, False, None, "a", 1.0, 7.0, 2),
        RequestSample(1, 0, 0.02, 200, True, False, None, "b", 3.0, 7.0, 2),
    )
    load_result = LoadTestResult(
        requested=2,
        successful_requests=2,
        failed_requests=0,
        http_429_responses=0,
        timed_out_requests=0,
        duration_seconds=0.02,
        throughput_requests_per_second=100.0,
        mean_latency_ms=15.0,
        p50_latency_ms=15.0,
        p95_latency_ms=19.5,
        p99_latency_ms=19.9,
        samples=samples,
    )
    before = {
        "completed_requests": 10,
        "backend_inference_calls": 3,
        "batches_executed": 3,
        "mean_backend_inference_ms": 4.0,
        "mean_realized_batch_size": 1.5,
    }
    after = {
        "completed_requests": 12,
        "backend_inference_calls": 4,
        "batches_executed": 4,
        "mean_backend_inference_ms": 5.0,
        "mean_realized_batch_size": 2.0,
        "backend": "torch",
        "device": "cpu",
    }

    result = benchmark_module._case_result(case, load_result, before, after)

    assert result.server_stats_delta["completed_requests"] == 2
    assert result.backend_inference_call_count == 1
    assert result.mean_queue_wait_ms == pytest.approx(2.0)
    assert result.p95_queue_wait_ms == pytest.approx(2.9)
    assert result.mean_backend_inference_ms == pytest.approx(8.0)
    assert result.mean_realized_batch_size == pytest.approx(3.5)
    assert result.maximum_realized_batch_size == 2


def test_benchmark_numeric_helpers_and_case_fallbacks() -> None:
    assert benchmark_module._numeric({"value": True}, "value") is None
    assert benchmark_module._numeric({"value": "3"}, "value") is None
    assert benchmark_module._integer({"value": False}, "value") == 0
    assert benchmark_module._mean_delta(
        {"count": 1, "mean": 2.0},
        {"count": 1, "mean": 2.0},
        mean_key="mean",
        count_key="count",
    ) is None
    assert benchmark_module._percentile([], 95) is None

    case = build_primary_cases(_settings())[0]
    samples = (
        RequestSample(0, 0, 0.01, 500, False, False, "failed", None, 99.0, 8.0, 9),
        RequestSample(1, 0, 0.02, 200, True, False, None, "ok", None, 6.0, None),
    )
    load_result = LoadTestResult(
        requested=2,
        successful_requests=1,
        failed_requests=1,
        http_429_responses=0,
        timed_out_requests=0,
        duration_seconds=0.02,
        throughput_requests_per_second=50.0,
        mean_latency_ms=20.0,
        p50_latency_ms=20.0,
        p95_latency_ms=20.0,
        p99_latency_ms=20.0,
        samples=samples,
    )
    result = benchmark_module._case_result(
        case,
        load_result,
        {"backend_inference_calls": 2, "batches_executed": 2},
        {"backend_inference_calls": 3, "batches_executed": 3},
    )

    assert result.mean_backend_inference_ms == pytest.approx(6.0)
    assert result.mean_realized_batch_size == pytest.approx(1.0)
    assert result.mean_queue_wait_ms is None
    assert result.maximum_realized_batch_size == 0
    assert result.backend == case.backend
    assert result.device == "unknown"


def test_loopback_socket_settings_manifest_and_environment_are_local_only() -> None:
    settings = _settings()
    updated = benchmark_module._settings_with_port(settings, 43210)
    assert updated.server.host == "127.0.0.1"
    assert updated.server.port == 43210

    sock = benchmark_module._open_loopback_socket()
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert int(port) > 0
        assert sock.getblocking() is False
    finally:
        sock.close()

    images = generate_synthetic_images(2)
    manifest = benchmark_module._image_manifest(images)
    assert [entry["sequence_index"] for entry in manifest] == [0, 1]
    assert all(len(str(entry["sha256"])) == 64 for entry in manifest)

    environment = benchmark_module._environment_metadata()
    assert environment["python_version"]
    assert "onnxruntime_available_providers" in environment
    assert isinstance(environment["cuda_available"], bool)


def test_parity_json_loader_and_hash_gate_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parity_path = tmp_path / "parity.json"
    parity_path.write_text('{"passed": true, "onnx_sha256": "abc"}', encoding="utf-8")
    checked: list[tuple[Path, Path, str]] = []

    def require(
        parity: str | Path,
        onnx: str | Path,
        *,
        expected_device: str = "cpu",
    ) -> object:
        checked.append((Path(parity), Path(onnx), expected_device))
        return object()

    monkeypatch.setattr("inference_gateway.parity.require_passing_parity", require)

    payload = benchmark_module._require_parity(_settings(), parity_path)
    cuda_settings = GatewaySettings.model_validate(
        {"model": {"device": "cuda"}, "benchmark": {"requests_per_case": 8}}
    )
    benchmark_module._require_parity(cuda_settings, parity_path)

    assert payload["passed"] is True
    assert checked == [
        (parity_path, Path("artifacts/resnet18.onnx"), "cpu"),
        (parity_path, Path("artifacts/resnet18.onnx"), "cuda"),
    ]

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        benchmark_module._read_json_object(invalid)


@pytest.mark.asyncio
async def test_readiness_waiter_handles_ready_and_failed_server_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadyResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, bool]:
            return {"ready": True}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def get(self, url: str) -> ReadyResponse:
            assert url.endswith("/readyz")
            return ReadyResponse()

    monkeypatch.setattr(benchmark_module.httpx, "AsyncClient", FakeClient)
    class FakeProcess:
        def __init__(self, *, alive: bool, exitcode: int | None) -> None:
            self.alive = alive
            self.exitcode = exitcode

        def is_alive(self) -> bool:
            return self.alive

    await benchmark_module._wait_until_ready(
        "http://gateway.test",
        FakeProcess(alive=True, exitcode=None),  # type: ignore[arg-type]
        timeout_seconds=1,
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        await benchmark_module._wait_until_ready(
            "http://gateway.test",
            FakeProcess(alive=False, exitcode=7),  # type: ignore[arg-type]
            timeout_seconds=1,
        )


@pytest.mark.asyncio
async def test_loopback_server_runs_in_a_spawned_process_with_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = GatewaySettings.model_validate(
        {
            "model": {"backend": "fake", "device": "cpu"},
            "scheduler": {"mode": "direct", "request_timeout_ms": 5000},
        }
    )
    original_create = benchmark_module._create_server_process
    created_processes: list[Any] = []

    def track_process(settings_arg: GatewaySettings, sock: object) -> tuple[Any, Any]:
        process, stop_signal = original_create(settings_arg, sock)  # type: ignore[arg-type]
        created_processes.append(process)
        return process, stop_signal

    monkeypatch.setattr(benchmark_module, "_create_server_process", track_process)

    async with benchmark_module._loopback_server(
        settings,
        startup_timeout_seconds=30,
    ) as base_url:
        assert len(created_processes) == 1
        assert created_processes[0].pid is not None
        assert created_processes[0].pid != os.getpid()
        async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
            health = await client.get("/healthz")
            ready = await client.get("/readyz")
        assert health.json() == {"status": "ok"}
        assert ready.json()["ready"] is True


@pytest.mark.asyncio
async def test_bounded_process_teardown_escalates_after_clean_signal() -> None:
    class FakeSignal:
        def __init__(self) -> None:
            self.was_set = False

        def is_set(self) -> bool:
            return self.was_set

        def set(self) -> None:
            self.was_set = True

        def wait(self, timeout: float | None = None) -> bool:
            del timeout
            return self.was_set

    class StubbornProcess:
        def __init__(self) -> None:
            self.alive = True
            self.exitcode: int | None = None
            self.joins: list[float | None] = []
            self.terminated = False
            self.killed = False
            self.closed = False

        def start(self) -> None:
            raise AssertionError("already started")

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.joins.append(timeout)
            if self.killed:
                self.alive = False
                self.exitcode = -9

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def close(self) -> None:
            self.closed = True

    process = StubbornProcess()
    signal = FakeSignal()

    exitcode = await benchmark_module._stop_server_process(process, signal)

    assert signal.was_set
    assert process.terminated
    assert process.killed
    assert process.joins == [30.0, 5.0, 5.0]
    assert process.closed
    assert exitcode == -9


@pytest.mark.asyncio
async def test_run_case_uses_one_generator_for_warmup_stats_and_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    case = build_primary_cases(_settings())[0]
    images = generate_synthetic_images(1)
    load_result = LoadTestResult(
        requested=case.requests,
        successful_requests=case.requests,
        failed_requests=0,
        http_429_responses=0,
        timed_out_requests=0,
        duration_seconds=2.0,
        throughput_requests_per_second=100.0,
        mean_latency_ms=2.0,
        p50_latency_ms=2.0,
        p95_latency_ms=3.0,
        p99_latency_ms=4.0,
        samples=(),
    )

    class FakeGenerator:
        def __init__(
            self,
            base_url: str,
            *,
            concurrency: int,
            timeout_seconds: float,
        ) -> None:
            events.append((base_url, concurrency, timeout_seconds))
            self.stats_calls = 0

        async def __aenter__(self) -> FakeGenerator:
            events.append("enter")
            return self

        async def __aexit__(self, *args: object) -> None:
            del args
            events.append("exit")

        async def warmup(self, received_images: object, requests: int) -> None:
            assert received_images is images
            events.append(("warmup", requests))

        async def get_json(self, path: str) -> dict[str, Any]:
            assert path == "/stats"
            self.stats_calls += 1
            return {
                "completed_requests": 0 if self.stats_calls == 1 else case.requests,
                "backend_inference_calls": 0 if self.stats_calls == 1 else case.requests,
                "batches_executed": 0 if self.stats_calls == 1 else case.requests,
                "backend": "torch",
                "device": "cpu",
            }

        async def run(self, received_images: object, requests: int) -> LoadTestResult:
            assert received_images is images
            assert requests == case.requests
            events.append("run")
            return load_result

    monkeypatch.setattr(benchmark_module, "AsyncLoadGenerator", FakeGenerator)

    result = await benchmark_module._run_case(
        case,
        base_url="http://gateway.test",
        images=images,
        warmup_requests=7,
        request_timeout_ms=0,
    )

    assert events == [
        ("http://gateway.test", 1, 5.0),
        "enter",
        ("warmup", 7),
        "run",
        "exit",
    ]
    assert result.backend_inference_call_count == case.requests


@pytest.mark.asyncio
async def test_parity_gate_runs_before_any_benchmark_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def fail_parity(settings: GatewaySettings, parity_path: Path) -> dict[str, Any]:
        del settings, parity_path
        events.append("parity")
        raise RuntimeError("parity rejected")

    @asynccontextmanager
    async def forbidden_server(
        settings: GatewaySettings,
    ) -> AsyncIterator[str]:
        del settings
        events.append("server")
        yield "http://never.test"

    monkeypatch.setattr(benchmark_module, "_require_parity", fail_parity)
    monkeypatch.setattr(benchmark_module, "_loopback_server", forbidden_server)

    with pytest.raises(RuntimeError, match="parity rejected"):
        await run_benchmark(_settings(tmp_path))

    assert events == ["parity"]


@pytest.mark.asyncio
async def test_benchmark_reuses_one_image_sequence_across_all_32_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server_settings: list[GatewaySettings] = []
    image_identities: list[int] = []
    written: list[tuple[Mapping[str, Any], Path]] = []
    sealed: list[Mapping[str, Any]] = []

    def pass_parity(settings: GatewaySettings, parity_path: Path) -> dict[str, Any]:
        del settings, parity_path
        return {"passed": True, "onnx_sha256": "a" * 64}

    @asynccontextmanager
    async def fake_server(settings: GatewaySettings) -> AsyncIterator[str]:
        server_settings.append(settings)
        yield "http://gateway.test"

    async def fake_run_case(
        case: BenchmarkCase,
        *,
        base_url: str,
        images: object,
        warmup_requests: int,
        request_timeout_ms: float,
    ) -> BenchmarkCaseResult:
        assert base_url == "http://gateway.test"
        assert warmup_requests == 25
        assert request_timeout_ms == 5000.0
        image_identities.append(id(images))
        return _case_result(case, index=len(image_identities))

    def fake_write(
        results: Mapping[str, Any],
        output_directory: str | Path,
        *,
        update_readme: bool,
        readme_path: str | Path,
    ) -> None:
        del update_readme, readme_path
        written.append((results, Path(output_directory)))

    def fake_seal(results: Mapping[str, Any]) -> dict[str, Any]:
        sealed.append(results)
        return dict(results)

    monkeypatch.setattr(benchmark_module, "_require_parity", pass_parity)
    monkeypatch.setattr(benchmark_module, "_loopback_server", fake_server)
    monkeypatch.setattr(benchmark_module, "_run_case", fake_run_case)
    monkeypatch.setattr(benchmark_module, "_environment_metadata", lambda: {"os": "test"})
    monkeypatch.setattr(reporting_module, "seal_benchmark_results", fake_seal)
    monkeypatch.setattr(reporting_module, "write_benchmark_artifacts", fake_write)

    result = await run_benchmark(_settings(tmp_path))

    assert len(server_settings) == 10
    assert len(result.cases) == 32
    assert len(set(image_identities)) == 1
    assert len(result.image_sequence) == 8
    assert sealed[0]["cases"][0]["case_id"] == result.cases[0].case_id
    assert written[0][0]["cases"][0]["case_id"] == result.cases[0].case_id
    assert written[0][1] == tmp_path / "artifacts"


def test_all_artifacts_come_from_full_precision_results(tmp_path: Path) -> None:
    results = _synthetic_results(tmp_path)
    output = tmp_path / "generated"

    paths = write_benchmark_artifacts(results, output)

    loaded = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert loaded["cases"][0]["throughput_requests_per_second"] == 123.12345678901234
    assert loaded["cases"][0]["samples"][0]["elapsed_seconds"] == (
        1.2345678901234567 / 1000.0
    )
    frame = pd.read_csv(paths.results_csv)
    assert len(frame) == 32
    assert "stats_delta_backend_inference_calls" in frame.columns
    assert frame.loc[0, "throughput_requests_per_second"] == pytest.approx(
        123.12345678901234
    )
    report = paths.report_markdown.read_text(encoding="utf-8")
    assert "onnx-dynamic-b16-w3-c64" in report
    assert "999.988 successful requests/s" in report
    assert "torch-direct-c1" in report
    assert "1.235 ms" in report
    assert "changed throughput by" in report
    assert "PyTorch versus ONNX" in report
    assert "| Success | Failures | Reject | Timeout |" in report
    assert "Mean realized batch | Max realized batch | Calls |" in report
    assert "once in a fixed case order" in report
    assert "Thermal throttling" in report
    assert paths.throughput_vs_p95_chart.read_bytes().startswith(b"\x89PNG")
    assert paths.batch_efficiency_chart.read_bytes().startswith(b"\x89PNG")


def test_best_cases_require_full_success_and_plot_groups_include_device() -> None:
    unsealed = deepcopy(_synthetic_results())
    unsealed.pop("provenance")
    partial_outlier = unsealed["cases"][-1]
    _set_case_success_count(
        partial_outlier,
        partial_outlier["requests"] - 1,
        throughput=1_000_000.0,
        latency_ms=0.0001,
    )
    results = seal_benchmark_results(unsealed)

    report = render_report(results)
    groups = reporting_module._group_plot_points(
        results["cases"],
        x_key="p95_latency_ms",
        y_key="throughput_requests_per_second",
    )

    assert "Best measured throughput: `onnx-dynamic-b16-w3-c32`" in report
    assert "Best measured p95 latency: `torch-direct-c1`" in report
    assert "1,000,000.000 successful requests/s" not in report
    assert "0.000 ms" not in report
    assert "torch / direct / cpu" in groups
    assert "onnx / dynamic / cpu" in groups


def test_comparisons_only_match_cases_on_the_same_device() -> None:
    cases = _synthetic_results()["cases"]
    torch_direct = dict(cases[1])
    onnx_direct = dict(cases[5])
    torch_dynamic = dict(cases[8])
    assert torch_direct["concurrency"] == onnx_direct["concurrency"] == 8
    assert torch_dynamic["concurrency"] == 8

    torch_direct["device"] = "cpu"
    onnx_direct["device"] = "cuda"
    torch_dynamic["device"] = "cuda"
    assert reporting_module._backend_comparison_lines(
        [torch_direct, onnx_direct]
    ) == [
        "- No matched fully successful PyTorch/ONNX direct measurements were available."
    ]
    assert reporting_module._batching_tradeoff_lines(
        [torch_direct, torch_dynamic]
    ) == ["- No matched fully successful direct/dynamic measurements were available."]

    onnx_direct["device"] = "cpu"
    torch_dynamic["device"] = "cpu"
    backend_lines = reporting_module._backend_comparison_lines(
        [torch_direct, onnx_direct]
    )
    batching_lines = reporting_module._batching_tradeoff_lines(
        [torch_direct, torch_dynamic]
    )
    assert backend_lines[0].startswith("- Direct cpu concurrency 8")
    assert batching_lines[0].startswith("- torch on cpu at concurrency 8")


def test_marked_readme_is_updated_from_the_same_results(tmp_path: Path) -> None:
    results = _synthetic_results(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Keep me\n\n"
        f"{README_RESULTS_START}\nold fabricated result\n{README_RESULTS_END}\n\n"
        "Keep this footer.\n",
        encoding="utf-8",
    )

    write_benchmark_artifacts(
        results,
        tmp_path / "artifacts",
        update_readme=True,
        readme_path=readme,
    )

    updated = readme.read_text(encoding="utf-8")
    assert updated.count(README_RESULTS_START) == 1
    assert updated.count(README_RESULTS_END) == 1
    assert "old fabricated result" not in updated
    assert "# Keep me" in updated
    assert "Keep this footer." in updated
    assert "999.988 req/s" in updated
    assert "Operating system: `TestOS 1.0`" in updated
    assert "Machine: `test-machine`" in updated
    assert "Processor: `test-cpu`" in updated
    assert "Logical CPUs: `8`" in updated
    assert "Python: `3.11.9`" in updated
    assert "artifacts/throughput_vs_p95.png" in updated
    assert "artifacts/report.md" in updated


@pytest.mark.parametrize(
    "contents",
    [
        "# no markers\n",
        f"{README_RESULTS_END}\nwrong order\n{README_RESULTS_START}\n",
        f"{README_RESULTS_START}\nmissing end\n",
    ],
)
def test_marked_readme_rejects_missing_or_misordered_markers(
    tmp_path: Path,
    contents: str,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="must contain ordered"):
        update_marked_readme(readme, _synthetic_results(), tmp_path / "artifacts")


def test_report_regeneration_and_result_validation(tmp_path: Path) -> None:
    results = _synthetic_results(tmp_path)
    source = tmp_path / "source" / "results.json"
    source.parent.mkdir()
    source.write_text(json.dumps(results), encoding="utf-8")

    paths = generate_report(source, output_directory=tmp_path / "regenerated")

    assert load_results(paths.results_json)["schema_version"] == 1
    assert paths.results_csv.is_file()
    assert paths.report_markdown.is_file()

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_results(array_path)

    missing_cases = tmp_path / "missing-cases.json"
    missing_cases.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark results fields"):
        load_results(missing_cases)

    malformed_case = tmp_path / "malformed-case.json"
    malformed_case.write_text('{"cases": [3]}', encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark results fields"):
        load_results(malformed_case)


def test_strict_validation_rejects_incomplete_duplicate_and_malformed_results() -> None:
    results = _synthetic_results()

    wrong_version = deepcopy(results)
    wrong_version["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported benchmark results schema"):
        validate_benchmark_results(wrong_version)

    partial = deepcopy(results)
    partial["cases"].pop()
    with pytest.raises(ValueError, match="exactly 32"):
        validate_benchmark_results(partial)

    duplicate = deepcopy(results)
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_benchmark_results(duplicate)

    nonfinite = deepcopy(results)
    nonfinite["cases"][0]["throughput_requests_per_second"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_benchmark_results(nonfinite)

    invalid_counts = deepcopy(results)
    invalid_counts["cases"][0]["successful_requests"] -= 1
    with pytest.raises(ValueError, match="counts do not sum"):
        validate_benchmark_results(invalid_counts)


def test_strict_validation_binds_config_parity_and_payload_provenance() -> None:
    results = _synthetic_results()
    validated = validate_benchmark_results(results)

    assert validated is results
    assert results["provenance"]["algorithm"] == "sha256-canonical-json-v1"
    assert all(
        len(results["provenance"][key]) == 64
        for key in (
            "payload_sha256",
            "benchmark_config_sha256",
            "parity_artifact_sha256",
        )
    )

    wrong_device = deepcopy(results)
    wrong_device["parity_artifact"]["device"] = "cuda"
    with pytest.raises(ValueError, match="parity_artifact is invalid"):
        validate_benchmark_results(wrong_device)

    lax_tolerance = deepcopy(results)
    lax_tolerance["parity_artifact"]["rtol"] = 1e-2
    with pytest.raises(ValueError, match="parity_artifact is invalid"):
        validate_benchmark_results(lax_tolerance)

    wrong_logit_width = deepcopy(results)
    wrong_logit_width["parity_artifact"]["per_batch"][1]["logit_count"] = 4004
    with pytest.raises(ValueError, match="parity_artifact is invalid"):
        validate_benchmark_results(wrong_logit_width)

    hand_edited = deepcopy(results)
    hand_edited["generated_at_utc"] = "2026-08-27T12:35:56+00:00"
    with pytest.raises(ValueError, match=r"provenance\.payload_sha256"):
        validate_benchmark_results(hand_edited)


def test_invalid_results_cannot_write_artifacts_or_update_readme(tmp_path: Path) -> None:
    invalid = deepcopy(_synthetic_results())
    invalid["cases"].pop()
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="exactly 32"):
        write_benchmark_artifacts(invalid, output)
    assert not output.exists()

    readme = tmp_path / "README.md"
    original = (
        f"# Preserve me\n\n{README_RESULTS_START}\nold\n{README_RESULTS_END}\n"
    )
    readme.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 32"):
        update_marked_readme(readme, invalid, output)
    assert readme.read_text(encoding="utf-8") == original


def test_report_handles_a_run_without_successful_measurements() -> None:
    unsealed = deepcopy(_synthetic_results())
    unsealed.pop("provenance")
    for failed_case in unsealed["cases"]:
        _set_case_success_count(failed_case, 0)
    results = seal_benchmark_results(unsealed)

    report = render_report(results)

    assert "No fully successful measurements were available" in report
    assert "No matched fully successful direct/dynamic measurements" in report
    assert "No matched fully successful PyTorch/ONNX direct measurements" in report


def test_sync_benchmark_adapter_forwards_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    expected = BenchmarkSuiteResult(
        schema_version=1,
        generated_at_utc="now",
        environment={},
        benchmark_config={},
        parity_artifact={"passed": True},
        image_sequence=(),
        cases=(),
    )

    async def fake_run(
        settings: GatewaySettings,
        *,
        parity_path: str | Path | None,
        update_readme: bool,
        readme_path: str | Path,
    ) -> BenchmarkSuiteResult:
        captured.update(
            {
                "settings": settings,
                "parity_path": parity_path,
                "update_readme": update_readme,
                "readme_path": readme_path,
            }
        )
        return expected

    monkeypatch.setattr(benchmark_module, "run_benchmark", fake_run)
    settings = _settings()

    actual = run_benchmark_sync(
        settings,
        parity_path="custom-parity.json",
        update_readme=True,
        readme_path="CUSTOM.md",
    )

    assert actual is expected
    assert captured == {
        "settings": settings,
        "parity_path": "custom-parity.json",
        "update_readme": True,
        "readme_path": "CUSTOM.md",
    }

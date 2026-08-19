from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
    update_marked_readme,
    write_benchmark_artifacts,
)


def _settings(tmp_path: Path | None = None) -> GatewaySettings:
    payload: dict[str, Any] = {}
    if tmp_path is not None:
        payload["output"] = {"directory": str(tmp_path / "artifacts")}
    return GatewaySettings.model_validate(payload)


def _case_result(
    case: BenchmarkCase,
    *,
    index: int = 0,
    throughput: float | None = None,
    p95_latency_ms: float | None = None,
) -> BenchmarkCaseResult:
    measured_throughput = throughput if throughput is not None else 100.0 + index
    measured_p95 = p95_latency_ms if p95_latency_ms is not None else 20.0 + index
    batch_size = 1.0 if case.scheduler_mode == "direct" else float(case.max_batch_size / 2)
    calls = case.requests if case.scheduler_mode == "direct" else max(1, case.requests // 4)
    sample = RequestSample(
        sequence=0,
        image_index=0,
        elapsed_seconds=0.12345678901234566,
        status_code=200,
        success=True,
        timed_out=False,
        error=None,
        request_id=f"request-{index}",
        queue_wait_ms=1.25,
        backend_inference_ms=4.5,
        realized_batch_size=int(batch_size),
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
        mean_latency_ms=10.123456789012345,
        p50_latency_ms=9.5,
        p95_latency_ms=measured_p95,
        p99_latency_ms=30.25,
        mean_queue_wait_ms=1.25,
        p95_queue_wait_ms=2.0,
        mean_backend_inference_ms=4.5,
        mean_realized_batch_size=batch_size,
        maximum_realized_batch_size=max(1, int(batch_size)),
        backend_inference_call_count=calls,
        batches_executed=calls,
        duration_seconds=2.345678901234567,
        server_stats_delta={
            "completed_requests": case.requests,
            "backend_inference_calls": calls,
        },
        server_stats_before={"completed_requests": 3},
        server_stats_after={"completed_requests": case.requests + 3},
        samples=(sample,),
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
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-08-27T12:34:56+00:00",
        "environment": {
            "operating_system": "TestOS 1.0",
            "machine": "test-machine",
            "processor": "test-cpu",
            "logical_cpu_count": 8,
            "python_version": "3.11.9",
            "torch_version": "test",
            "torchvision_version": "test",
            "onnx_version": "test",
            "onnxruntime_version": "test",
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_name": None,
        },
        "benchmark_config": settings.model_dump(mode="json"),
        "parity_artifact": {
            "passed": True,
            "onnx_sha256": "a" * 64,
            "max_abs_difference": 1e-6,
            "mean_abs_difference": 1e-7,
            "top1_agreement": 1.0,
        },
        "image_sequence": [
            {
                "sequence_index": 0,
                "filename": "synthetic-000.png",
                "media_type": "image/png",
                "byte_length": 100,
                "sha256": "b" * 64,
            }
        ],
        "cases": [result.to_dict() for result in case_results],
    }


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
    checked: list[tuple[Path, Path]] = []

    def require(parity: str | Path, onnx: str | Path) -> object:
        checked.append((Path(parity), Path(onnx)))
        return object()

    monkeypatch.setattr("inference_gateway.parity.require_passing_parity", require)

    payload = benchmark_module._require_parity(_settings(), parity_path)

    assert payload["passed"] is True
    assert checked == [(parity_path, Path("artifacts/resnet18.onnx"))]

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
    pending = asyncio.create_task(asyncio.sleep(10))
    try:
        await benchmark_module._wait_until_ready(
            "http://gateway.test",
            pending,
            timeout_seconds=1,
        )
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    async def fail() -> None:
        raise OSError("startup failed")

    failed = asyncio.create_task(fail())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="failed during startup") as captured:
        await benchmark_module._wait_until_ready(
            "http://gateway.test",
            failed,
            timeout_seconds=1,
        )
    assert isinstance(captured.value.__cause__, OSError)


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

    monkeypatch.setattr(benchmark_module, "_require_parity", pass_parity)
    monkeypatch.setattr(benchmark_module, "_loopback_server", fake_server)
    monkeypatch.setattr(benchmark_module, "_run_case", fake_run_case)
    monkeypatch.setattr(benchmark_module, "_environment_metadata", lambda: {"os": "test"})
    monkeypatch.setattr(reporting_module, "write_benchmark_artifacts", fake_write)

    result = await run_benchmark(_settings(tmp_path))

    assert len(server_settings) == 10
    assert len(result.cases) == 32
    assert len(set(image_identities)) == 1
    assert len(result.image_sequence) == 8
    assert written[0][0]["cases"][0]["case_id"] == result.cases[0].case_id
    assert written[0][1] == tmp_path / "artifacts"


def test_all_artifacts_come_from_full_precision_results(tmp_path: Path) -> None:
    results = _synthetic_results(tmp_path)
    output = tmp_path / "generated"

    paths = write_benchmark_artifacts(results, output)

    loaded = json.loads(paths.results_json.read_text(encoding="utf-8"))
    assert loaded["cases"][0]["throughput_requests_per_second"] == 123.12345678901234
    assert loaded["cases"][0]["samples"][0]["elapsed_seconds"] == 0.12345678901234566
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
    assert paths.throughput_vs_p95_chart.read_bytes().startswith(b"\x89PNG")
    assert paths.batch_efficiency_chart.read_bytes().startswith(b"\x89PNG")


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
    with pytest.raises(ValueError, match="non-empty cases"):
        load_results(missing_cases)

    malformed_case = tmp_path / "malformed-case.json"
    malformed_case.write_text('{"cases": [3]}', encoding="utf-8")
    with pytest.raises(ValueError, match="case 0"):
        load_results(malformed_case)


def test_report_handles_a_run_without_successful_measurements() -> None:
    results = _synthetic_results()
    failed_case = dict(results["cases"][0])
    failed_case.update(
        {
            "successful_requests": 0,
            "throughput_requests_per_second": None,
            "p95_latency_ms": None,
            "mean_realized_batch_size": None,
        }
    )
    results["cases"] = [failed_case]

    report = render_report(results)

    assert "No successful measurements were available" in report
    assert "No matched successful direct/dynamic measurements" in report
    assert "No matched successful PyTorch/ONNX direct measurements" in report


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

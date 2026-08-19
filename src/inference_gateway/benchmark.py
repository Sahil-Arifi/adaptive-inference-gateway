"""Reproducible 32-case benchmark orchestration over real loopback HTTP."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import uvicorn

from inference_gateway.config import BackendName, GatewaySettings, SchedulerMode
from inference_gateway.loadgen import (
    AsyncLoadGenerator,
    LoadTestResult,
    RequestSample,
    SyntheticImage,
    generate_synthetic_images,
)

PRIMARY_CASE_COUNT = 32

_COUNTER_FIELDS = (
    "total_accepted_requests",
    "completed_requests",
    "rejected_requests",
    "timed_out_requests",
    "cancelled_requests",
    "failed_requests",
    "backend_inference_calls",
    "batches_executed",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Configuration for one measured point in the primary matrix."""

    case_id: str
    backend: str
    scheduler_mode: str
    concurrency: int
    max_batch_size: int
    max_wait_ms: float
    requests: int

    @property
    def server_group(self) -> tuple[str, str, int, float]:
        """Cases in one group can share a loaded backend and server."""

        return (
            self.backend,
            self.scheduler_mode,
            self.max_batch_size,
            self.max_wait_ms,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    """One benchmark result, including raw samples and server counter deltas."""

    case_id: str
    backend: str
    device: str
    scheduler_mode: str
    concurrency: int
    max_batch_size: int
    max_wait_ms: float
    requests: int
    successful_requests: int
    failures: int
    rejections: int
    timeouts: int
    throughput_requests_per_second: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    mean_queue_wait_ms: float | None
    p95_queue_wait_ms: float | None
    mean_backend_inference_ms: float | None
    mean_realized_batch_size: float | None
    maximum_realized_batch_size: int
    backend_inference_call_count: int
    batches_executed: int
    duration_seconds: float
    server_stats_delta: dict[str, int | float]
    server_stats_before: dict[str, Any]
    server_stats_after: dict[str, Any]
    samples: tuple[RequestSample, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result without rounding raw values."""

        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Metadata and all primary benchmark cases."""

    schema_version: int
    generated_at_utc: str
    environment: dict[str, Any]
    benchmark_config: dict[str, Any]
    parity_artifact: dict[str, Any]
    image_sequence: tuple[dict[str, Any], ...]
    cases: tuple[BenchmarkCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this suite for the canonical results.json artifact."""

        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "environment": self.environment,
            "benchmark_config": self.benchmark_config,
            "parity_artifact": self.parity_artifact,
            "image_sequence": list(self.image_sequence),
            "cases": [case.to_dict() for case in self.cases],
        }


def build_primary_cases(settings: GatewaySettings) -> tuple[BenchmarkCase, ...]:
    """Build the required eight direct and twenty-four dynamic cases."""

    benchmark = settings.benchmark
    cases: list[BenchmarkCase] = []
    for backend in (BackendName.TORCH, BackendName.ONNX):
        for concurrency in benchmark.concurrency:
            cases.append(
                BenchmarkCase(
                    case_id=f"{backend.value}-direct-c{concurrency}",
                    backend=backend.value,
                    scheduler_mode=SchedulerMode.DIRECT.value,
                    concurrency=concurrency,
                    max_batch_size=1,
                    max_wait_ms=0.0,
                    requests=benchmark.requests_per_case,
                )
            )

    for backend in (BackendName.TORCH, BackendName.ONNX):
        for max_batch_size in benchmark.dynamic_batch_sizes:
            for max_wait_ms in benchmark.dynamic_wait_ms:
                for concurrency in benchmark.dynamic_concurrency:
                    wait_token = format(max_wait_ms, "g").replace(".", "p")
                    cases.append(
                        BenchmarkCase(
                            case_id=(
                                f"{backend.value}-dynamic-b{max_batch_size}-"
                                f"w{wait_token}-c{concurrency}"
                            ),
                            backend=backend.value,
                            scheduler_mode=SchedulerMode.DYNAMIC.value,
                            concurrency=concurrency,
                            max_batch_size=max_batch_size,
                            max_wait_ms=max_wait_ms,
                            requests=benchmark.requests_per_case,
                        )
                    )

    case_ids = {case.case_id for case in cases}
    if len(cases) != PRIMARY_CASE_COUNT or len(case_ids) != PRIMARY_CASE_COUNT:
        raise ValueError(
            "primary benchmark matrix must contain exactly 32 unique cases "
            f"(built {len(cases)} cases, {len(case_ids)} unique IDs)"
        )
    return tuple(cases)


def group_primary_cases(
    cases: Sequence[BenchmarkCase],
) -> tuple[tuple[BenchmarkCase, ...], ...]:
    """Group concurrency variants to avoid reloading a model for every case."""

    grouped: dict[tuple[str, str, int, float], list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[case.server_group].append(case)
    groups = tuple(tuple(group) for group in grouped.values())
    # 2 direct backend groups + 2 backends * 2 batch sizes * 2 waits.
    if len(groups) != 10:
        raise ValueError(f"expected 10 practical server groups, found {len(groups)}")
    return groups


def _numeric(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _stats_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, int | float]:
    delta: dict[str, int | float] = {}
    for key in _COUNTER_FIELDS:
        delta[key] = _integer(after, key) - _integer(before, key)
    delta["current_queue_depth"] = _integer(after, "current_queue_depth")
    return delta


def _mean_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    mean_key: str,
    count_key: str,
) -> float | None:
    """Recover a per-case mean from cumulative server means and counts."""

    count_before = _integer(before, count_key)
    count_after = _integer(after, count_key)
    count_delta = count_after - count_before
    mean_before = _numeric(before, mean_key)
    mean_after = _numeric(after, mean_key)
    if count_delta <= 0 or mean_after is None:
        return None
    total_after = mean_after * count_after
    total_before = (mean_before or 0.0) * count_before
    return (total_after - total_before) / count_delta


def _sample_values(
    samples: Sequence[RequestSample],
    attribute: str,
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        if not sample.success:
            continue
        candidate = getattr(sample, attribute)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            values.append(float(candidate))
    return values


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _case_result(
    case: BenchmarkCase,
    load_result: LoadTestResult,
    before: dict[str, Any],
    after: dict[str, Any],
) -> BenchmarkCaseResult:
    delta = _stats_delta(before, after)
    queue_waits = _sample_values(load_result.samples, "queue_wait_ms")
    backend_durations = _sample_values(load_result.samples, "backend_inference_ms")
    realized_batch_sizes = _sample_values(load_result.samples, "realized_batch_size")

    backend_call_count = int(delta.get("backend_inference_calls", 0))
    batches_executed = int(delta.get("batches_executed", 0))
    mean_backend_duration = _mean_delta(
        before,
        after,
        mean_key="mean_backend_inference_ms",
        count_key="backend_inference_calls",
    )
    if mean_backend_duration is None and backend_durations:
        mean_backend_duration = float(np.mean(backend_durations))
    mean_batch_size = _mean_delta(
        before,
        after,
        mean_key="mean_realized_batch_size",
        count_key="batches_executed",
    )
    if mean_batch_size is None and batches_executed > 0:
        mean_batch_size = load_result.successful_requests / batches_executed

    actual_backend = after.get("backend")
    actual_device = after.get("device")
    return BenchmarkCaseResult(
        case_id=case.case_id,
        backend=actual_backend if isinstance(actual_backend, str) else case.backend,
        device=actual_device if isinstance(actual_device, str) else "unknown",
        scheduler_mode=case.scheduler_mode,
        concurrency=case.concurrency,
        max_batch_size=case.max_batch_size,
        max_wait_ms=case.max_wait_ms,
        requests=case.requests,
        successful_requests=load_result.successful_requests,
        failures=load_result.failed_requests,
        rejections=load_result.http_429_responses,
        timeouts=load_result.timed_out_requests,
        throughput_requests_per_second=load_result.throughput_requests_per_second,
        mean_latency_ms=load_result.mean_latency_ms,
        p50_latency_ms=load_result.p50_latency_ms,
        p95_latency_ms=load_result.p95_latency_ms,
        p99_latency_ms=load_result.p99_latency_ms,
        mean_queue_wait_ms=float(np.mean(queue_waits)) if queue_waits else None,
        p95_queue_wait_ms=_percentile(queue_waits, 95.0),
        mean_backend_inference_ms=mean_backend_duration,
        mean_realized_batch_size=mean_batch_size,
        maximum_realized_batch_size=(
            int(max(realized_batch_sizes)) if realized_batch_sizes else 0
        ),
        backend_inference_call_count=backend_call_count,
        batches_executed=batches_executed,
        duration_seconds=load_result.duration_seconds,
        server_stats_delta=delta,
        server_stats_before=before,
        server_stats_after=after,
        samples=load_result.samples,
    )


def _settings_for_group(
    base: GatewaySettings,
    first_case: BenchmarkCase,
    port: int,
) -> GatewaySettings:
    model = base.model.model_copy(update={"backend": BackendName(first_case.backend)})
    scheduler = base.scheduler.model_copy(
        update={
            "mode": SchedulerMode(first_case.scheduler_mode),
            "max_batch_size": first_case.max_batch_size,
            "max_wait_ms": first_case.max_wait_ms,
        }
    )
    server = base.server.model_copy(update={"host": "127.0.0.1", "port": port})
    return base.model_copy(
        update={"model": model, "scheduler": scheduler, "server": server},
        deep=True,
    )


def _settings_with_port(settings: GatewaySettings, port: int) -> GatewaySettings:
    server = settings.server.model_copy(update={"host": "127.0.0.1", "port": port})
    return settings.model_copy(update={"server": server}, deep=True)


def _open_loopback_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(2048)
    sock.setblocking(False)
    return sock


async def _wait_until_ready(
    base_url: str,
    server_task: asyncio.Task[None],
    *,
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        while asyncio.get_running_loop().time() < deadline:
            if server_task.done():
                exception = server_task.exception()
                if exception is not None:
                    raise RuntimeError("benchmark server failed during startup") from exception
                raise RuntimeError("benchmark server exited before becoming ready")
            try:
                response = await client.get(f"{base_url}/readyz")
                if response.status_code == 200:
                    payload: Any = response.json()
                    if isinstance(payload, dict) and payload.get("ready") is True:
                        return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.05)
    raise TimeoutError(f"benchmark server did not become ready within {timeout_seconds}s")


@asynccontextmanager
async def _loopback_server(
    settings: GatewaySettings,
    *,
    startup_timeout_seconds: float = 300.0,
) -> AsyncIterator[str]:
    # Import lazily so matrix construction/report tests need not initialize FastAPI.
    from inference_gateway.service import create_app

    sock = _open_loopback_socket()
    address = sock.getsockname()
    if not isinstance(address, tuple) or len(address) < 2:
        sock.close()
        raise RuntimeError("could not determine loopback server address")
    port = int(address[1])
    server_settings = _settings_with_port(settings, port)
    app = create_app(server_settings)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve(sockets=[sock]), name="benchmark-uvicorn")
    base_url = f"http://127.0.0.1:{port}"
    try:
        await _wait_until_ready(
            base_url,
            server_task,
            timeout_seconds=startup_timeout_seconds,
        )
        yield base_url
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=60.0)
        except TimeoutError:
            server.force_exit = True
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        finally:
            sock.close()
def _environment_metadata() -> dict[str, Any]:
    def package_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    metadata: dict[str, Any] = {
        "operating_system": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "torch_version": package_version("torch"),
        "torchvision_version": package_version("torchvision"),
        "onnx_version": package_version("onnx"),
        "onnxruntime_version": package_version("onnxruntime"),
    }
    try:
        import onnxruntime as ort

        metadata["onnxruntime_available_providers"] = ort.get_available_providers()
    except ImportError:
        metadata["onnxruntime_available_providers"] = []
    try:
        import torch

        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_version"] = torch.version.cuda
        metadata["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        metadata["cuda_available"] = False
        metadata["cuda_version"] = None
        metadata["cuda_device_name"] = None
    return metadata


def _image_manifest(images: Sequence[SyntheticImage]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "sequence_index": index,
            "filename": image.filename,
            "media_type": image.media_type,
            "byte_length": len(image.content),
            "sha256": hashlib.sha256(image.content).hexdigest(),
        }
        for index, image in enumerate(images)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _require_parity(settings: GatewaySettings, parity_path: Path) -> dict[str, Any]:
    # This helper validates pass status and binds the report to the exact ONNX hash.
    from inference_gateway.parity import require_passing_parity

    require_passing_parity(parity_path, settings.model.onnx_path)
    return _read_json_object(parity_path)


async def _run_case(
    case: BenchmarkCase,
    *,
    base_url: str,
    images: Sequence[SyntheticImage],
    warmup_requests: int,
    request_timeout_ms: float,
) -> BenchmarkCaseResult:
    timeout_seconds = max(5.0, request_timeout_ms / 1000.0 + 5.0)
    async with AsyncLoadGenerator(
        base_url,
        concurrency=case.concurrency,
        timeout_seconds=timeout_seconds,
    ) as generator:
        await generator.warmup(images, warmup_requests)
        before = await generator.get_json("/stats")
        load_result = await generator.run(images, case.requests)
        after = await generator.get_json("/stats")
    return _case_result(case, load_result, before, after)


async def run_benchmark(
    settings: GatewaySettings,
    *,
    parity_path: str | Path | None = None,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> BenchmarkSuiteResult:
    """Run all 32 CPU-primary cases and write every required artifact.

    A passing parity report whose ONNX hash matches the configured model is a
    mandatory preflight. Servers are reused by backend/scheduler configuration,
    but each concurrency point receives its own warmup and stats baseline.
    """

    cases = build_primary_cases(settings)
    groups = group_primary_cases(cases)
    output_directory = settings.output.directory
    parity_report_path = (
        Path(parity_path) if parity_path is not None else output_directory / "parity.json"
    )
    parity_artifact = _require_parity(settings, parity_report_path)
    images = generate_synthetic_images(settings.benchmark.synthetic_image_count)

    results: list[BenchmarkCaseResult] = []
    for group in groups:
        first_case = group[0]
        group_settings = _settings_for_group(settings, first_case, settings.server.port)
        async with _loopback_server(group_settings) as base_url:
            for case in group:
                result = await _run_case(
                    case,
                    base_url=base_url,
                    images=images,
                    warmup_requests=settings.benchmark.warmup_requests,
                    request_timeout_ms=settings.scheduler.request_timeout_ms,
                )
                results.append(result)

    if len(results) != PRIMARY_CASE_COUNT:
        raise RuntimeError(f"benchmark completed {len(results)} of {PRIMARY_CASE_COUNT} cases")

    suite = BenchmarkSuiteResult(
        schema_version=1,
        generated_at_utc=datetime.now(UTC).isoformat(),
        environment=_environment_metadata(),
        benchmark_config=settings.model_dump(mode="json"),
        parity_artifact=parity_artifact,
        image_sequence=_image_manifest(images),
        cases=tuple(results),
    )

    from inference_gateway.reporting import write_benchmark_artifacts

    write_benchmark_artifacts(
        suite.to_dict(),
        output_directory,
        update_readme=update_readme,
        readme_path=readme_path,
    )
    return suite


def run_benchmark_sync(
    settings: GatewaySettings,
    *,
    parity_path: str | Path | None = None,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> BenchmarkSuiteResult:
    """Synchronous CLI adapter for :func:`run_benchmark`."""

    return asyncio.run(
        run_benchmark(
            settings,
            parity_path=parity_path,
            update_readme=update_readme,
            readme_path=readme_path,
        )
    )


__all__ = [
    "PRIMARY_CASE_COUNT",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkSuiteResult",
    "build_primary_cases",
    "group_primary_cases",
    "run_benchmark",
    "run_benchmark_sync",
]

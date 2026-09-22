"""Deterministic image generation and asynchronous HTTP load generation."""

from __future__ import annotations

import asyncio
import io
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class SyntheticImage:
    """A deterministic, encoded benchmark image."""

    filename: str
    media_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class RequestSample:
    """Raw outcome and server telemetry for one measured HTTP request."""

    sequence: int
    image_index: int
    elapsed_seconds: float
    status_code: int | None
    success: bool
    timed_out: bool
    error: str | None
    request_id: str | None
    queue_wait_ms: float | None
    backend_inference_ms: float | None
    realized_batch_size: int | None
    server_processing_ms: float | None = None
    upload_and_parse_ms: float | None = None
    preprocessing_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without rounding timings."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadTestResult:
    """Aggregate measurements plus the individual samples they came from."""

    requested: int
    successful_requests: int
    failed_requests: int
    http_429_responses: int
    timed_out_requests: int
    duration_seconds: float
    throughput_requests_per_second: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    samples: tuple[RequestSample, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a full-precision JSON-serializable result."""

        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload


def generate_synthetic_images(
    count: int = 8,
    *,
    width: int = 256,
    height: int = 256,
    seed: int = 20260827,
) -> tuple[SyntheticImage, ...]:
    """Generate a repeatable mix of valid RGB PNG and JPEG images locally.

    The image bytes are created once before a load test and reused. Encoding and
    image generation are therefore excluded from request latency measurements.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")

    yy, xx = np.indices((height, width), dtype=np.uint32)
    images: list[SyntheticImage] = []
    for index in range(count):
        # Structured patterns exercise decoding without depending on a dataset.
        rng = np.random.default_rng(seed + index)
        noise = rng.integers(0, 32, size=(height, width), dtype=np.uint8)
        red = ((xx * (index + 3) + yy + noise) % 256).astype(np.uint8)
        green = ((yy * (index + 5) + xx // 2 + noise) % 256).astype(np.uint8)
        blue = (((xx + yy) * (index + 7) + noise) % 256).astype(np.uint8)
        pixels = np.stack((red, green, blue), axis=-1)
        image = Image.fromarray(pixels, mode="RGB")

        image_format = "PNG" if index % 2 == 0 else "JPEG"
        suffix = image_format.lower().replace("jpeg", "jpg")
        media_type = "image/png" if image_format == "PNG" else "image/jpeg"
        buffer = io.BytesIO()
        if image_format == "PNG":
            image.save(buffer, format=image_format, compress_level=6)
        else:
            image.save(
                buffer,
                format=image_format,
                quality=90,
                optimize=False,
                progressive=False,
                subsampling=0,
            )
        images.append(
            SyntheticImage(
                filename=f"synthetic-{index:03d}.{suffix}",
                media_type=media_type,
                content=buffer.getvalue(),
            )
        )
    return tuple(images)


def write_deterministic_image(path: str | Path, *, index: int = 0) -> Path:
    """Write one deterministic image for the CLI demo without external data."""

    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("demo image path must end in .png, .jpg, or .jpeg")
    # Select an even source for PNG and an odd source for JPEG.
    source_index = index * 2 if suffix == ".png" else index * 2 + 1
    image = generate_synthetic_images(source_index + 1)[source_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image.content)
    return output_path


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class AsyncLoadGenerator:
    """One-client asynchronous load generator with a fixed worker pool."""

    def __init__(
        self,
        base_url: str,
        *,
        concurrency: int,
        timeout_seconds: float = 30.0,
        file_field: str = "file",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._concurrency = concurrency
        self._file_field = file_field
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            ),
            transport=transport,
        )
        self._closed = False

    async def __aenter__(self) -> AsyncLoadGenerator:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the sole underlying HTTP client."""

        if not self._closed:
            self._closed = True
            await self._client.aclose()

    async def get_json(self, path: str) -> dict[str, Any]:
        """Fetch a successful JSON object using the load-test client."""

        response = await self._client.get(path)
        response.raise_for_status()
        payload: Any = response.json()
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise ValueError(f"{path} did not return a JSON object")
        return payload

    async def warmup(
        self,
        images: Sequence[SyntheticImage],
        requests: int,
    ) -> None:
        """Send unmeasured warmups and require every response to succeed."""

        if requests < 0:
            raise ValueError("warmup requests cannot be negative")
        if requests == 0:
            return
        result = await self.run(images, requests)
        if result.successful_requests != requests:
            raise RuntimeError(
                "warmup failed: "
                f"{result.successful_requests}/{requests} requests succeeded"
            )

    async def run(
        self,
        images: Sequence[SyntheticImage],
        requests: int,
    ) -> LoadTestResult:
        """Run measured requests with at most ``concurrency`` in flight."""

        if self._closed:
            raise RuntimeError("load generator is closed")
        if requests < 1:
            raise ValueError("requests must be at least 1")
        if not images:
            raise ValueError("at least one image is required")

        samples: list[RequestSample | None] = [None] * requests
        next_sequence = iter(range(requests))

        async def worker() -> None:
            for sequence in next_sequence:
                image_index = sequence % len(images)
                samples[sequence] = await self._send_one(
                    sequence=sequence,
                    image_index=image_index,
                    image=images[image_index],
                )

        started = time.perf_counter()
        workers = [
            asyncio.create_task(worker(), name=f"loadgen-worker-{index}")
            for index in range(min(self._concurrency, requests))
        ]
        await asyncio.gather(*workers)
        duration_seconds = time.perf_counter() - started

        completed_samples = tuple(sample for sample in samples if sample is not None)
        if len(completed_samples) != requests:
            raise RuntimeError("load generator lost request samples")
        successful = sum(sample.success for sample in completed_samples)
        http_429 = sum(sample.status_code == 429 for sample in completed_samples)
        timed_out = sum(sample.timed_out for sample in completed_samples)
        successful_latencies_ms = [
            sample.elapsed_seconds * 1000.0
            for sample in completed_samples
            if sample.success
        ]
        return LoadTestResult(
            requested=requests,
            successful_requests=successful,
            failed_requests=requests - successful,
            http_429_responses=http_429,
            timed_out_requests=timed_out,
            duration_seconds=duration_seconds,
            throughput_requests_per_second=(
                successful / duration_seconds if duration_seconds > 0 else 0.0
            ),
            mean_latency_ms=(
                statistics.fmean(successful_latencies_ms)
                if successful_latencies_ms
                else None
            ),
            p50_latency_ms=_percentile(successful_latencies_ms, 50.0),
            p95_latency_ms=_percentile(successful_latencies_ms, 95.0),
            p99_latency_ms=_percentile(successful_latencies_ms, 99.0),
            samples=completed_samples,
        )

    async def _send_one(
        self,
        *,
        sequence: int,
        image_index: int,
        image: SyntheticImage,
    ) -> RequestSample:
        started = time.perf_counter()
        status_code: int | None = None
        try:
            response = await self._client.post(
                "/v1/predict",
                files={
                    self._file_field: (
                        image.filename,
                        image.content,
                        image.media_type,
                    )
                },
            )
            elapsed = time.perf_counter() - started
            status_code = response.status_code
            success = 200 <= response.status_code < 300
            payload: Mapping[str, Any] = {}
            try:
                candidate: Any = response.json()
                if isinstance(candidate, dict):
                    payload = candidate
            except ValueError:
                pass
            request_id = payload.get("request_id")
            return RequestSample(
                sequence=sequence,
                image_index=image_index,
                elapsed_seconds=elapsed,
                status_code=status_code,
                success=success,
                timed_out=response.status_code in {408, 504},
                error=None if success else f"HTTP {response.status_code}",
                request_id=request_id if isinstance(request_id, str) else None,
                queue_wait_ms=_optional_float(payload, "queue_wait_ms"),
                backend_inference_ms=_optional_float(payload, "backend_inference_ms"),
                realized_batch_size=_optional_int(payload, "realized_batch_size"),
                server_processing_ms=_optional_float(payload, "server_processing_ms"),
                upload_and_parse_ms=_optional_float(payload, "upload_and_parse_ms"),
                preprocessing_ms=_optional_float(payload, "preprocessing_ms"),
            )
        except httpx.TimeoutException as exc:
            return RequestSample(
                sequence=sequence,
                image_index=image_index,
                elapsed_seconds=time.perf_counter() - started,
                status_code=status_code,
                success=False,
                timed_out=True,
                error=type(exc).__name__,
                request_id=None,
                queue_wait_ms=None,
                backend_inference_ms=None,
                realized_batch_size=None,
            )
        except httpx.HTTPError as exc:
            return RequestSample(
                sequence=sequence,
                image_index=image_index,
                elapsed_seconds=time.perf_counter() - started,
                status_code=status_code,
                success=False,
                timed_out=False,
                error=type(exc).__name__,
                request_id=None,
                queue_wait_ms=None,
                backend_inference_ms=None,
                realized_batch_size=None,
            )


async def run_load_test(
    base_url: str,
    *,
    requests: int,
    concurrency: int,
    warmup_requests: int = 0,
    images: Sequence[SyntheticImage] | None = None,
    timeout_seconds: float = 30.0,
    file_field: str = "file",
    transport: httpx.AsyncBaseTransport | None = None,
) -> LoadTestResult:
    """Convenience API that performs warmup and measurement with one client."""

    benchmark_images = tuple(images) if images is not None else generate_synthetic_images()
    async with AsyncLoadGenerator(
        base_url,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        file_field=file_field,
        transport=transport,
    ) as generator:
        await generator.warmup(benchmark_images, warmup_requests)
        return await generator.run(benchmark_images, requests)


__all__ = [
    "AsyncLoadGenerator",
    "LoadTestResult",
    "RequestSample",
    "SyntheticImage",
    "generate_synthetic_images",
    "run_load_test",
    "write_deterministic_image",
]

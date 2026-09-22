from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from inference_gateway.loadgen import (
    AsyncLoadGenerator,
    RequestSample,
    generate_synthetic_images,
    run_load_test,
    write_deterministic_image,
)


def test_synthetic_images_are_deterministic_valid_rgb_and_alternate_formats() -> None:
    first = generate_synthetic_images(6, width=48, height=32, seed=17)
    second = generate_synthetic_images(6, width=48, height=32, seed=17)

    assert first == second
    assert [image.media_type for image in first] == [
        "image/png",
        "image/jpeg",
        "image/png",
        "image/jpeg",
        "image/png",
        "image/jpeg",
    ]
    assert [image.filename.rsplit(".", 1)[-1] for image in first] == [
        "png",
        "jpg",
        "png",
        "jpg",
        "png",
        "jpg",
    ]
    for expected_format, encoded in zip(
        ("PNG", "JPEG", "PNG", "JPEG", "PNG", "JPEG"),
        first,
        strict=True,
    ):
        with Image.open(io.BytesIO(encoded.content)) as decoded:
            assert decoded.format == expected_format
            assert decoded.mode == "RGB"
            assert decoded.size == (48, 32)
            decoded.verify()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count must be at least 1"),
        ({"width": 0}, "dimensions must be positive"),
        ({"height": 0}, "dimensions must be positive"),
    ],
)
def test_synthetic_image_validation(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_synthetic_images(**kwargs)


def test_write_deterministic_image_supports_png_and_jpeg(tmp_path: Path) -> None:
    png = write_deterministic_image(tmp_path / "nested" / "demo.png")
    jpeg = write_deterministic_image(tmp_path / "demo.jpeg", index=1)

    with Image.open(png) as decoded_png, Image.open(jpeg) as decoded_jpeg:
        assert decoded_png.format == "PNG"
        assert decoded_jpeg.format == "JPEG"
    with pytest.raises(ValueError, match="must end in"):
        write_deterministic_image(tmp_path / "demo.gif")


@pytest.mark.asyncio
async def test_concurrency_is_bounded_warmups_are_excluded_and_telemetry_is_parsed() -> None:
    active = 0
    maximum_active = 0
    call_count = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active, call_count
        assert request.url.path == "/v1/predict"
        async with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            call_index = call_count
            call_count += 1
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return httpx.Response(
            200,
            json={
                "request_id": f"request-{call_index}",
                "queue_wait_ms": 1.25,
                "backend_inference_ms": 2,
                "realized_batch_size": 3,
                "server_processing_ms": 8.5,
                "upload_and_parse_ms": 1.0,
                "preprocessing_ms": 2.0,
            },
        )

    result = await run_load_test(
        "http://gateway.test",
        requests=7,
        concurrency=3,
        warmup_requests=2,
        images=generate_synthetic_images(2),
        transport=httpx.MockTransport(handler),
    )

    assert call_count == 9
    assert maximum_active == 3
    assert result.requested == 7
    assert result.successful_requests == 7
    assert [sample.sequence for sample in result.samples] == list(range(7))
    assert [sample.image_index for sample in result.samples] == [0, 1, 0, 1, 0, 1, 0]
    assert all(sample.request_id is not None for sample in result.samples)
    assert {sample.queue_wait_ms for sample in result.samples} == {1.25}
    assert {sample.backend_inference_ms for sample in result.samples} == {2.0}
    assert {sample.realized_batch_size for sample in result.samples} == {3}
    assert {sample.server_processing_ms for sample in result.samples} == {8.5}
    assert {sample.upload_and_parse_ms for sample in result.samples} == {1.0}
    assert {sample.preprocessing_ms for sample in result.samples} == {2.0}


@pytest.mark.asyncio
async def test_percentiles_are_computed_from_individual_successful_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latencies = (0.001, 0.002, 0.003, 9.0)

    async def fake_send(
        self: AsyncLoadGenerator,
        *,
        sequence: int,
        image_index: int,
        image: object,
    ) -> RequestSample:
        del self, image
        success = sequence < 3
        return RequestSample(
            sequence=sequence,
            image_index=image_index,
            elapsed_seconds=latencies[sequence],
            status_code=200 if success else 500,
            success=success,
            timed_out=False,
            error=None if success else "HTTP 500",
            request_id=f"r-{sequence}" if success else None,
            queue_wait_ms=None,
            backend_inference_ms=None,
            realized_batch_size=None,
        )

    monkeypatch.setattr(AsyncLoadGenerator, "_send_one", fake_send)
    async with AsyncLoadGenerator(
        "http://gateway.test",
        concurrency=2,
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    ) as generator:
        result = await generator.run(generate_synthetic_images(1), 4)

    measured_ms = np.asarray([1.0, 2.0, 3.0])
    assert result.mean_latency_ms == pytest.approx(2.0)
    assert result.p50_latency_ms == pytest.approx(np.percentile(measured_ms, 50))
    assert result.p95_latency_ms == pytest.approx(np.percentile(measured_ms, 95))
    assert result.p99_latency_ms == pytest.approx(np.percentile(measured_ms, 99))
    assert result.failed_requests == 1


@pytest.mark.asyncio
async def test_status_timeout_and_transport_error_accounting() -> None:
    outcomes: list[int | str] = [200, 429, 504, 500, "timeout", "connect"]
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        outcome = outcomes[call_count]
        call_count += 1
        if outcome == "timeout":
            raise httpx.ReadTimeout("late", request=request)
        if outcome == "connect":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(int(outcome), json={"detail": "synthetic"})

    result = await run_load_test(
        "http://gateway.test",
        requests=len(outcomes),
        concurrency=2,
        transport=httpx.MockTransport(handler),
    )

    assert result.successful_requests == 1
    assert result.failed_requests == 5
    assert result.http_429_responses == 1
    assert result.timed_out_requests == 2
    assert [sample.status_code for sample in result.samples] == [200, 429, 504, 500, None, None]
    assert result.samples[1].error == "HTTP 429"
    assert result.samples[4].error == "ReadTimeout"
    assert result.samples[5].error == "ConnectError"


@pytest.mark.asyncio
async def test_invalid_or_non_json_telemetry_is_ignored() -> None:
    responses = iter(
        (
            httpx.Response(
                200,
                json={
                    "request_id": 12,
                    "queue_wait_ms": True,
                    "backend_inference_ms": "slow",
                    "realized_batch_size": 2.5,
                },
            ),
            httpx.Response(200, content=b"not-json"),
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return next(responses)

    result = await run_load_test(
        "http://gateway.test",
        requests=2,
        concurrency=1,
        transport=httpx.MockTransport(handler),
    )

    for sample in result.samples:
        assert sample.request_id is None
        assert sample.queue_wait_ms is None
        assert sample.backend_inference_ms is None
        assert sample.realized_batch_size is None


@pytest.mark.asyncio
async def test_generator_validation_json_fetch_failure_and_close_are_safe() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        AsyncLoadGenerator("http://gateway.test", concurrency=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        AsyncLoadGenerator("http://gateway.test", concurrency=1, timeout_seconds=0)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/array":
            return httpx.Response(200, json=[1, 2, 3])
        return httpx.Response(500)

    generator = AsyncLoadGenerator(
        "http://gateway.test",
        concurrency=1,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="JSON object"):
        await generator.get_json("/array")
    with pytest.raises(httpx.HTTPStatusError):
        await generator.get_json("/error")
    with pytest.raises(ValueError, match="requests must be"):
        await generator.run(generate_synthetic_images(1), 0)
    with pytest.raises(ValueError, match="at least one image"):
        await generator.run((), 1)
    with pytest.raises(ValueError, match="cannot be negative"):
        await generator.warmup(generate_synthetic_images(1), -1)
    await generator.warmup(generate_synthetic_images(1), 0)
    await generator.close()
    await generator.close()
    with pytest.raises(RuntimeError, match="closed"):
        await generator.run(generate_synthetic_images(1), 1)


@pytest.mark.asyncio
async def test_warmup_failure_is_reported_and_empty_success_has_no_percentiles() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503)

    async with AsyncLoadGenerator(
        "http://gateway.test",
        concurrency=1,
        transport=httpx.MockTransport(handler),
    ) as generator:
        with pytest.raises(RuntimeError, match="warmup failed"):
            await generator.warmup(generate_synthetic_images(1), 1)
        result = await generator.run(generate_synthetic_images(1), 2)

    assert result.throughput_requests_per_second == 0.0
    assert result.mean_latency_ms is None
    assert result.p50_latency_ms is None
    assert result.p95_latency_ms is None
    assert result.p99_latency_ms is None
    assert result.to_dict()["samples"][0]["success"] is False

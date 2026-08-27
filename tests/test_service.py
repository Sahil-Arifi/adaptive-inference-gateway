from __future__ import annotations

import asyncio
import struct
import threading
import time
import zlib
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from io import BytesIO
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference_gateway import service
from inference_gateway.backends.fake_backend import FakeBackend
from inference_gateway.config import GatewaySettings
from inference_gateway.service import (
    BoundedPreprocessor,
    PredictionAdmissionController,
    PreprocessingQueueFullError,
    RequestBodyLimitMiddleware,
    create_app,
)


def make_settings(
    *,
    mode: str = "direct",
    max_batch_size: int = 16,
    max_wait_ms: float = 2.0,
    max_queue_size: int = 256,
    timeout_ms: float = 5000.0,
    max_upload_bytes: int = 1_000_000,
    max_in_flight_requests: int = 64,
    prometheus: bool = True,
) -> GatewaySettings:
    return GatewaySettings.model_validate(
        {
            "model": {"backend": "fake", "device": "cpu"},
            "server": {
                "max_upload_bytes": max_upload_bytes,
                "max_in_flight_requests": max_in_flight_requests,
                "top_k": 5,
            },
            "scheduler": {
                "mode": mode,
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "max_queue_size": max_queue_size,
                "request_timeout_ms": timeout_ms,
                "inference_workers": 1,
            },
            "observability": {"prometheus": prometheus},
        }
    )


def image_bytes(image_format: str = "PNG", mode: str = "RGB") -> bytes:
    image = Image.new(mode, (48, 32), color=127)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def oversized_png_header(width: int = 6000, height: int = 6000) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(name: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", checksum)

    return signature + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def multipart_request(payload: bytes | None = None) -> tuple[bytes, dict[str, str]]:
    boundary = "aig-admission-boundary"
    image = payload or image_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
    return body, {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }


class GatedMultipartStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, gate: asyncio.Event, *, delay_seconds: float = 0.0) -> None:
        self.body = body
        self.gate = gate
        self.delay_seconds = delay_seconds
        self.started = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started = True
        split = max(1, len(self.body) // 2)
        yield self.body[:split]
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        await self.gate.wait()
        yield self.body[split:]


class NeverFinishingMultipartStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.started = False
        self.cancelled = False
        self._never_complete = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started = True
        split = max(1, len(self.body) // 2)
        yield self.body[:split]
        try:
            await self._never_complete.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("never-finishing stream was unexpectedly released")


async def wait_for_admission_count(
    admission: PredictionAdmissionController,
    expected: int,
) -> None:
    def reached_expected_count() -> bool:
        deadline = time.monotonic() + 5
        while admission.in_flight != expected:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

    assert await asyncio.to_thread(reached_expected_count)


def test_health_readiness_prediction_stats_and_metrics() -> None:
    backend = FakeBackend()
    app = create_app(make_settings(), backend=backend)

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        readiness = client.get("/readyz")
        assert readiness.status_code == 200
        assert readiness.json() == {
            "ready": True,
            "backend_loaded": True,
            "warmup_completed": True,
            "scheduler_running": True,
        }

        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["request_id"]) == 32
        assert payload["backend"] == "fake"
        assert payload["device"] == "cpu"
        assert payload["scheduler_mode"] == "direct"
        assert payload["realized_batch_size"] == 1
        assert len(payload["predictions"]) == 5
        assert payload["top_prediction_index"] == payload["predictions"][0]["index"]

        stats = client.get("/stats").json()
        assert stats["total_accepted_requests"] == 1
        assert stats["completed_requests"] == 1
        assert stats["backend_inference_calls"] == 1
        assert stats["batches_executed"] == 1

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "text/plain" in metrics.headers["content-type"]
        assert "inference_gateway_requests_total 1.0" in metrics.text

    assert backend.warmed_up


@pytest.mark.parametrize(
    ("filename", "media_type", "payload"),
    [
        ("gray.jpg", "image/jpeg", image_bytes("JPEG", "L")),
        ("rgba.png", "image/png", image_bytes("PNG", "RGBA")),
    ],
)
def test_predict_accepts_jpeg_png_and_converts_to_rgb(
    filename: str,
    media_type: str,
    payload: bytes,
) -> None:
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": (filename, payload, media_type)},
        )

    assert response.status_code == 200


def test_predict_rejects_missing_file() -> None:
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post("/v1/predict")

    assert response.status_code == 422


def test_predict_rejects_unsupported_media_type() -> None:
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("image.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 415
    assert "only image/jpeg and image/png" in response.json()["detail"]


def test_predict_rejects_malformed_image_without_traceback() -> None:
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("broken.png", b"not an image", "image/png")},
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 422
    assert response.json() == {"detail": "The uploaded file is not a valid image."}
    assert "Traceback" not in response.text
    assert "inference_gateway_requests_total 1.0" in metrics
    assert "inference_gateway_request_failures_total 1.0" in metrics
    assert "inference_gateway_request_latency_seconds_count 1.0" in metrics


def test_predict_rejects_valid_but_mismatched_image_format() -> None:
    image = Image.new("RGB", (16, 16))
    buffer = BytesIO()
    image.save(buffer, format="GIF")
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("claimed.png", buffer.getvalue(), "image/png")},
        )

    assert response.status_code == 415
    assert "Unsupported image format" in response.json()["detail"]


def test_predict_enforces_upload_size_limit() -> None:
    with TestClient(
        create_app(make_settings(max_upload_bytes=8), backend=FakeBackend())
    ) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("large.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 413
    assert "8-byte upload limit" in response.json()["detail"]


def test_raw_body_limit_runs_before_multipart_parsing() -> None:
    with TestClient(
        create_app(make_settings(max_upload_bytes=8), backend=FakeBackend())
    ) as client:
        response = client.post(
            "/v1/predict",
            content=b"x" * 70_000,
            headers={"Content-Type": "application/octet-stream"},
        )

    assert response.status_code == 413
    assert "transport limit" in response.json()["detail"]


@pytest.mark.asyncio
async def test_streamed_body_without_content_length_is_still_bounded() -> None:
    messages = iter(
        (
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"789012", "more_body": False},
        )
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def consume_body(
        _scope: dict[str, Any],
        receive_message: Any,
        _send_message: Any,
    ) -> None:
        while True:
            message = await receive_message()
            if not message.get("more_body", False):
                return

    middleware = RequestBodyLimitMiddleware(consume_body, max_body_bytes=10)
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/predict",
        "headers": [],
    }

    await middleware(scope, receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_outer_admission_bounds_concurrent_multipart_and_releases_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_preprocess(_payload: bytes) -> np.ndarray:
        return np.zeros((3, 224, 224), dtype=np.float32)

    monkeypatch.setattr(service, "preprocess_image", fast_preprocess)
    app = create_app(make_settings(), backend=FakeBackend())
    admission = app.state.prediction_admission
    assert isinstance(admission, PredictionAdmissionController)
    assert admission.capacity == 64
    body, headers = multipart_request()
    release_uploads = asyncio.Event()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            accepted_streams = [
                GatedMultipartStream(body, release_uploads) for _ in range(admission.capacity)
            ]
            accepted_tasks = [
                asyncio.create_task(
                    client.post("/v1/predict", content=stream, headers=headers)
                )
                for stream in accepted_streams
            ]
            await wait_for_admission_count(admission, admission.capacity)

            rejected_gate = asyncio.Event()
            rejected_gate.set()
            rejected_stream = GatedMultipartStream(body, rejected_gate)
            rejected = await client.post(
                "/v1/predict",
                content=rejected_stream,
                headers=headers,
            )
            metrics_while_full = (await client.get("/metrics")).text
            stats_while_full = (await client.get("/stats")).json()

            assert rejected.status_code == 429
            assert rejected.headers["retry-after"] == "0"
            assert rejected.json() == {"detail": "prediction admission capacity is full"}
            assert not rejected_stream.started
            assert admission.in_flight == admission.capacity
            assert admission.maximum_observed == admission.capacity
            assert stats_while_full["rejected_requests"] == 1
            assert "inference_gateway_requests_total 65.0" in metrics_while_full
            assert "inference_gateway_request_failures_total 1.0" in metrics_while_full

            release_uploads.set()
            async with asyncio.timeout(15):
                completed = await asyncio.gather(*accepted_tasks)
            assert all(response.status_code == 200 for response in completed)
            assert admission.in_flight == 0

            reused = await client.post(
                "/v1/predict",
                files={"file": ("image.png", image_bytes(), "image/png")},
            )
            assert reused.status_code == 200
            assert admission.in_flight == 0

            cancellation_gate = asyncio.Event()
            cancelled_stream = GatedMultipartStream(body, cancellation_gate)
            cancelled_task = asyncio.create_task(
                client.post("/v1/predict", content=cancelled_stream, headers=headers)
            )
            await wait_for_admission_count(admission, 1)
            cancelled_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_task
            cancellation_gate.set()
            assert admission.in_flight == 0

    for task in accepted_tasks:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_arrival_deadline_includes_multipart_parsing_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    app = create_app(make_settings(timeout_ms=5), backend=backend)
    body, headers = multipart_request()
    release = asyncio.Event()
    release.set()
    delayed_stream = GatedMultipartStream(body, release, delay_seconds=0.03)
    preprocessor_called = False

    async with app.router.lifespan_context(app):
        def forbidden_submit(_payload: bytes) -> asyncio.Future[np.ndarray]:
            nonlocal preprocessor_called
            preprocessor_called = True
            raise AssertionError("expired upload must not enter preprocessing")

        monkeypatch.setattr(app.state.preprocessor, "submit", forbidden_submit)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/predict",
                content=delayed_stream,
                headers=headers,
            )
            stats = (await client.get("/stats")).json()

    assert response.status_code == 504
    assert response.json() == {"detail": "inference request exceeded its deadline"}
    assert not preprocessor_called
    assert backend.call_count == 0
    assert stats["total_accepted_requests"] == 0
    assert stats["backend_inference_calls"] == 0
    assert stats["timed_out_requests"] == 1
    assert app.state.prediction_admission.in_flight == 0


@pytest.mark.asyncio
async def test_never_finishing_multipart_times_out_and_slot_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_preprocess(_payload: bytes) -> np.ndarray:
        return np.zeros((3, 224, 224), dtype=np.float32)

    monkeypatch.setattr(service, "preprocess_image", fast_preprocess)
    app = create_app(make_settings(timeout_ms=200), backend=FakeBackend())
    admission = app.state.prediction_admission
    body, headers = multipart_request()
    stream = NeverFinishingMultipartStream(body)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = time.monotonic()
            response = await client.post(
                "/v1/predict",
                content=stream,
                headers=headers,
            )
            elapsed = time.monotonic() - started

            assert response.status_code == 504
            assert response.json() == {"detail": "inference request exceeded its deadline"}
            assert elapsed < 1.0
            assert stream.started
            assert stream.cancelled
            assert admission.in_flight == 0

            reused = await client.post(
                "/v1/predict",
                files={"file": ("image.png", image_bytes(), "image/png")},
            )
            stats = (await client.get("/stats")).json()
            metrics = (await client.get("/metrics")).text

    assert reused.status_code == 200
    assert admission.in_flight == 0
    assert stats["timed_out_requests"] == 1
    assert "inference_gateway_requests_total 2.0" in metrics
    assert "inference_gateway_request_failures_total 1.0" in metrics


@pytest.mark.asyncio
async def test_admission_timeout_never_starts_a_second_response() -> None:
    admission = PredictionAdmissionController(1)
    sent: list[dict[str, Any]] = []

    async def started_then_hangs(
        _scope: dict[str, Any],
        _receive: Any,
        send: Any,
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await asyncio.Event().wait()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = service.PredictionAdmissionMiddleware(
        started_then_hangs,
        admission=admission,
        request_timeout_ms=10,
    )
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/predict",
        "headers": [],
    }

    with pytest.raises(TimeoutError):
        await middleware(scope, receive, send)

    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 200
    assert admission.in_flight == 0


@pytest.mark.asyncio
async def test_inner_asgi_timeout_error_is_not_translated_to_deadline_response() -> None:
    admission = PredictionAdmissionController(1)
    sent: list[dict[str, Any]] = []

    async def inner_timeout(
        _scope: dict[str, Any],
        _receive: Any,
        _send: Any,
    ) -> None:
        raise TimeoutError("inner ASGI timeout")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = service.PredictionAdmissionMiddleware(
        inner_timeout,
        admission=admission,
        request_timeout_ms=1000,
    )
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/predict",
        "headers": [],
    }

    with pytest.raises(TimeoutError, match="inner ASGI timeout"):
        await middleware(scope, receive, send)

    assert sent == []
    assert admission.in_flight == 0


def test_declared_pixel_bomb_is_rejected_as_413() -> None:
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("oversized.png", oversized_png_header(), "image/png")},
        )

    assert response.status_code == 413
    assert "decoded limit" in response.json()["detail"]


def test_metrics_can_be_disabled() -> None:
    with TestClient(
        create_app(make_settings(prometheus=False), backend=FakeBackend())
    ) as client:
        response = client.get("/metrics")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_bounded_preprocessor_rejects_an_unbounded_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_preprocess(_payload: bytes) -> np.ndarray:
        entered.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test preprocessor was not released")
        return np.zeros((3, 224, 224), dtype=np.float32)

    monkeypatch.setattr(service, "preprocess_image", blocked_preprocess)
    preprocessor = BoundedPreprocessor(capacity=1, max_workers=1)
    first = preprocessor.submit(b"first")
    assert await asyncio.to_thread(entered.wait, 1)

    with pytest.raises(PreprocessingQueueFullError, match="queue is full"):
        preprocessor.submit(b"second")

    release.set()
    assert (await first).shape == (3, 224, 224)
    await preprocessor.close()


def test_http_preprocessing_capacity_rejection_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(make_settings(), backend=FakeBackend())
    with TestClient(app) as client:
        def reject(_payload: bytes) -> asyncio.Future[np.ndarray]:
            raise PreprocessingQueueFullError("full")

        monkeypatch.setattr(app.state.preprocessor, "submit", reject)
        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        stats = client.get("/stats").json()

    assert response.status_code == 429
    assert stats["rejected_requests"] == 1


def test_backend_exception_is_sanitized() -> None:
    with TestClient(
        create_app(make_settings(), backend=FakeBackend(fail=True)),
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}
    assert "Configured fake backend failure" not in response.text


class InnerTimeoutBackend(FakeBackend):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        raise TimeoutError("backend internal timeout")


def test_backend_timeout_error_is_an_execution_failure_not_a_deadline() -> None:
    with TestClient(create_app(make_settings(), backend=InnerTimeoutBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        stats = client.get("/stats").json()

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}
    assert "backend internal timeout" not in response.text
    assert stats["failed_requests"] == 1
    assert stats["timed_out_requests"] == 0
    assert stats["cancelled_requests"] == 0


def test_preprocessing_timeout_error_is_an_execution_failure_not_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inner_timeout(_payload: bytes) -> np.ndarray:
        raise TimeoutError("preprocessor internal timeout")

    monkeypatch.setattr(service, "preprocess_image", inner_timeout)
    with TestClient(create_app(make_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        stats = client.get("/stats").json()

    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}
    assert "preprocessor internal timeout" not in response.text
    assert stats["total_accepted_requests"] == 0
    assert stats["timed_out_requests"] == 0
    assert stats["cancelled_requests"] == 0


@pytest.mark.parametrize("_attempt", range(3))
def test_real_timeout_path_returns_504_and_settles_request(_attempt: int) -> None:
    settings = make_settings(
        mode="dynamic",
        max_batch_size=1,
        max_wait_ms=1,
        timeout_ms=10,
    )
    with TestClient(create_app(settings, backend=FakeBackend(delay_seconds=0.05))) as client:
        response = client.post(
            "/v1/predict",
            files={"file": ("image.png", image_bytes(), "image/png")},
        )
        stats = client.get("/stats").json()

    assert response.status_code == 504
    assert stats["timed_out_requests"] == 1
    assert stats["cancelled_requests"] == 0


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocking backend")
        return super().predict_logits(batch)


def test_queue_overflow_returns_http_429() -> None:
    backend = BlockingBackend()
    settings = make_settings(
        mode="dynamic",
        max_batch_size=1,
        max_wait_ms=0,
        max_queue_size=1,
        timeout_ms=4000,
    )
    app = create_app(settings, backend=backend)
    payload = image_bytes()

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post,
            "/v1/predict",
            files={"file": ("first.png", payload, "image/png")},
        )
        assert backend.entered.wait(timeout=2)

        second = pool.submit(
            client.post,
            "/v1/predict",
            files={"file": ("second.png", payload, "image/png")},
        )
        deadline = time.monotonic() + 2
        while app.state.runtime.stats.current_queue_depth != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert app.state.runtime.stats.current_queue_depth == 1

        rejected = client.post(
            "/v1/predict",
            files={"file": ("rejected.png", payload, "image/png")},
        )
        backend.release.set()
        assert first.result(timeout=5).status_code == 200
        assert second.result(timeout=5).status_code == 200
        stats = client.get("/stats").json()

    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "0"
    assert stats["rejected_requests"] == 1

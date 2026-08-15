from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference_gateway.backends.fake_backend import FakeBackend
from inference_gateway.config import GatewaySettings
from inference_gateway.service import create_app


def make_settings(
    *,
    mode: str = "direct",
    max_batch_size: int = 16,
    max_wait_ms: float = 2.0,
    max_queue_size: int = 256,
    timeout_ms: float = 5000.0,
    max_upload_bytes: int = 1_000_000,
    prometheus: bool = True,
) -> GatewaySettings:
    return GatewaySettings.model_validate(
        {
            "model": {"backend": "fake", "device": "cpu"},
            "server": {"max_upload_bytes": max_upload_bytes, "top_k": 5},
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

    assert response.status_code == 422
    assert response.json() == {"detail": "The uploaded file is not a valid image."}
    assert "Traceback" not in response.text


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


def test_metrics_can_be_disabled() -> None:
    with TestClient(
        create_app(make_settings(prometheus=False), backend=FakeBackend())
    ) as client:
        response = client.get("/metrics")

    assert response.status_code == 404


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


def test_real_timeout_path_returns_504_and_settles_request() -> None:
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

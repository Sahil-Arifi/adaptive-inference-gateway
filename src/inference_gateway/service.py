"""FastAPI application and thin HTTP-to-runtime translation layer."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status

from inference_gateway.backends import FakeBackend, InferenceBackend, OnnxBackend, TorchBackend
from inference_gateway.config import BackendName, GatewaySettings, load_config
from inference_gateway.metrics import GatewayMetrics
from inference_gateway.preprocessing import (
    ImageDecodeError,
    UnsupportedImageFormatError,
    imagenet_categories,
    preprocess_image,
)
from inference_gateway.queueing import (
    DeadlineExceededError,
    QueueFullError,
    SchedulerClosedError,
)
from inference_gateway.runtime import InferenceRuntime
from inference_gateway.schemas import HealthResponse, Prediction, PredictResponse, ReadinessResponse

LOGGER = logging.getLogger(__name__)
ACCEPTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png"})


def build_backend(settings: GatewaySettings) -> InferenceBackend:
    """Construct the configured synchronous backend before runtime startup."""

    backend_name = settings.model.backend
    device = settings.model.device.value
    if backend_name is BackendName.TORCH:
        return TorchBackend(device=device)
    if backend_name is BackendName.ONNX:
        if not settings.model.onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {settings.model.onnx_path}; "
                "run `inference-gateway export --config <path>` first"
            )
        return OnnxBackend(settings.model.onnx_path, device=device)
    if backend_name is BackendName.FAKE:
        return FakeBackend()
    raise ValueError(f"unsupported backend: {backend_name}")


def create_app(
    settings: GatewaySettings | None = None,
    *,
    backend: InferenceBackend | None = None,
) -> FastAPI:
    """Create an isolated gateway application.

    Tests inject ``FakeBackend`` so no production weight download or ONNX file
    is required. Production and benchmark callers omit it and use configuration.
    """

    app_settings = settings or load_config("configs/default.yaml")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        selected_backend = backend or build_backend(app_settings)
        metrics = GatewayMetrics(enabled=app_settings.observability.prometheus)
        runtime = InferenceRuntime(
            selected_backend,
            app_settings.scheduler,
            metrics=metrics,
        )
        app.state.runtime = runtime
        try:
            await runtime.start()
            yield
        finally:
            await runtime.close()

    application = FastAPI(
        title="Adaptive Inference Gateway",
        version="0.1.0",
        description="Bounded image inference with direct or dynamic scheduling.",
        lifespan=lifespan,
    )
    application.state.settings = app_settings

    @application.get("/healthz", response_model=HealthResponse, tags=["operations"])
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/readyz",
        response_model=ReadinessResponse,
        tags=["operations"],
        responses={503: {"description": "Runtime has not completed startup"}},
    )
    async def readyz(request: Request, response: Response) -> ReadinessResponse:
        runtime = _runtime(request)
        ready = runtime.ready
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            ready=ready,
            backend_loaded=runtime.backend_loaded,
            warmup_completed=runtime.warmup_completed,
            scheduler_running=runtime.scheduler_running,
        )

    @application.get("/stats", tags=["operations"])
    async def stats(request: Request) -> dict[str, int | float | str]:
        return _runtime(request).stats_snapshot()

    @application.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        runtime = _runtime(request)
        if not runtime.metrics.enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prometheus metrics are disabled",
            )
        return Response(
            content=runtime.metrics_payload(),
            headers={"Content-Type": runtime.metrics.content_type},
        )

    @application.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            415: {"description": "Unsupported upload media type or image format"},
            422: {"description": "Malformed image"},
            429: {"description": "Bounded request queue is full"},
            503: {"description": "Runtime is not ready"},
            504: {"description": "Request deadline exceeded"},
        },
    )
    async def predict(
        request: Request,
        file: Annotated[UploadFile, File(...)],
    ) -> PredictResponse:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + app_settings.scheduler.request_timeout_ms / 1000.0
        runtime = _runtime(request)
        if not runtime.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="inference runtime is not ready",
            )

        media_type = (file.content_type or "").lower()
        if media_type not in ACCEPTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="only image/jpeg and image/png uploads are accepted",
            )

        limit = app_settings.server.max_upload_bytes
        payload = await file.read(limit + 1)
        if len(payload) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"image exceeds the {limit}-byte upload limit",
            )

        request_id = uuid.uuid4().hex
        try:
            tensor = await asyncio.to_thread(preprocess_image, payload)
            result = await runtime.predict(
                tensor,
                request_id=request_id,
                deadline_monotonic=deadline,
            )
        except UnsupportedImageFormatError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except ImageDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except QueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="inference queue is full",
                headers={"Retry-After": "0"},
            ) from exc
        except DeadlineExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="inference request exceeded its deadline",
            ) from exc
        except SchedulerClosedError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="inference runtime is unavailable",
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("inference request %s failed", request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="inference failed",
            ) from exc

        predictions = _top_predictions(result.logits, app_settings.server.top_k)
        top = predictions[0]
        return PredictResponse(
            request_id=result.request_id,
            top_prediction_index=top.index,
            top_prediction_label=top.label,
            confidence=top.confidence,
            predictions=predictions,
            backend=runtime.backend.name,
            device=runtime.backend.device,
            scheduler_mode=runtime.stats.scheduler_mode,
            server_processing_ms=(time.perf_counter() - started) * 1000.0,
            queue_wait_ms=result.queue_wait_ms,
            backend_inference_ms=result.backend_inference_ms,
            realized_batch_size=result.realized_batch_size,
        )

    return application


def _runtime(request: Request) -> InferenceRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, InferenceRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference runtime has not started",
        )
    return runtime


def _top_predictions(logits: np.ndarray, top_k: int) -> list[Prediction]:
    row = np.asarray(logits, dtype=np.float64)
    if row.ndim != 1 or row.shape[0] != 1000 or not np.all(np.isfinite(row)):
        raise ValueError(f"backend must return one finite 1000-logit row; received {row.shape}")
    shifted = row - np.max(row)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    indices = np.argsort(probabilities)[::-1][:top_k]
    labels = imagenet_categories()
    return [
        Prediction(
            index=int(index),
            label=labels[int(index)],
            confidence=float(probabilities[int(index)]),
        )
        for index in indices
    ]


__all__ = ["ACCEPTED_MEDIA_TYPES", "build_backend", "create_app"]

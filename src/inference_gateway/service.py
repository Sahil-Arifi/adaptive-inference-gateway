"""FastAPI application and thin HTTP-to-runtime translation layer."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from inference_gateway.backends import FakeBackend, InferenceBackend, OnnxBackend, TorchBackend
from inference_gateway.config import BackendName, GatewaySettings, load_config
from inference_gateway.metrics import GatewayMetrics
from inference_gateway.preprocessing import (
    ImageDecodeError,
    ImageTooLargeError,
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
MULTIPART_OVERHEAD_ALLOWANCE_BYTES = 64 * 1024
PREDICTION_ARRIVAL_STATE_KEY = "prediction_arrival_monotonic"
PREDICTION_DEADLINE_STATE_KEY = "prediction_deadline_monotonic"
PREDICTION_TIMEOUT_RECORDED_STATE_KEY = "prediction_timeout_recorded"


class RequestBodyTooLargeError(RuntimeError):
    """Raised by the raw ASGI receive wrapper before multipart spooling can grow."""


class RequestDeadlineExpiredError(RuntimeError):
    """Raised only when this service's configured absolute deadline expires."""


class RequestBodyLimitMiddleware:
    """Bound `/v1/predict` request bodies before FastAPI parses multipart data."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/predict"
        ):
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await _body_too_large_response(self.max_body_bytes)(scope, receive, send)
            return

        consumed = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_body_bytes:
                    raise RequestBodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise
            await _body_too_large_response(self.max_body_bytes)(scope, receive, send)


class PredictionAdmissionController:
    """Process-local, non-waiting admission counter shared with ASGI middleware."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("prediction admission capacity must be positive")
        self.capacity = capacity
        self._in_flight = 0
        self._maximum_observed = 0
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def maximum_observed(self) -> int:
        with self._lock:
            return self._maximum_observed

    def try_acquire(self) -> bool:
        """Acquire immediately or return false without queuing a waiter."""

        with self._lock:
            if self._in_flight >= self.capacity:
                return False
            self._in_flight += 1
            self._maximum_observed = max(self._maximum_observed, self._in_flight)
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight < 1:
                raise RuntimeError("prediction admission token released without acquisition")
            self._in_flight -= 1


class PredictionAdmissionMiddleware:
    """Reject excess prediction requests before multipart parsing or spooling."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        admission: PredictionAdmissionController,
        request_timeout_ms: float,
    ) -> None:
        if request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
        self.app = app
        self.admission = admission
        self.request_timeout_seconds = request_timeout_ms / 1000.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_prediction_request(scope):
            await self.app(scope, receive, send)
            return

        arrival = time.monotonic()
        deadline = arrival + self.request_timeout_seconds
        loop = asyncio.get_running_loop()
        remaining = max(deadline - time.monotonic(), 0.0)
        loop_deadline = loop.time() + remaining
        scope_state = scope.setdefault("state", {})
        scope_state[PREDICTION_ARRIVAL_STATE_KEY] = arrival
        scope_state[PREDICTION_DEADLINE_STATE_KEY] = deadline

        if not self.admission.try_acquire():
            runtime = _runtime_from_scope(scope)
            if runtime is not None:
                runtime.stats.record_rejected()
                runtime.metrics.record_queue_rejection()
            await _admission_rejected_response()(scope, receive, send)
            return

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            deadline_timeout = asyncio.timeout_at(loop_deadline)
            try:
                async with deadline_timeout:
                    await self.app(scope, receive, tracked_send)
            except TimeoutError:
                if not deadline_timeout.expired():
                    raise
                runtime = _runtime_from_scope(scope)
                _record_timeout_once(scope, runtime)
                if response_started:
                    raise
                await _deadline_exceeded_response()(scope, receive, tracked_send)
        finally:
            self.admission.release()


class PredictionMetricsMiddleware:
    """Measure every prediction request across validation, preprocessing, and inference."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_prediction_request(scope):
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        runtime = _runtime_from_scope(scope)
        if runtime is not None:
            runtime.metrics.record_request()

        async def capture_status(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            runtime = runtime or _runtime_from_scope(scope)
            if runtime is not None:
                runtime.metrics.observe_request_latency(
                    (time.perf_counter() - started) * 1000.0
                )
                if response_status >= status.HTTP_400_BAD_REQUEST:
                    runtime.metrics.record_failure()


class PreprocessingQueueFullError(RuntimeError):
    """Raised when bounded image preprocessing admission is exhausted."""


class BoundedPreprocessor:
    """Run image transforms off-loop without an unbounded executor backlog."""

    def __init__(self, *, capacity: int, max_workers: int) -> None:
        if capacity < 1 or max_workers < 1:
            raise ValueError("preprocessing capacity and workers must be positive")
        self.capacity = capacity
        self.max_workers = max_workers
        self._pending = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="image-preprocessor",
        )

    @property
    def pending(self) -> int:
        return self._pending

    def submit(self, payload: bytes) -> asyncio.Future[np.ndarray]:
        if self._closed:
            raise SchedulerClosedError("image preprocessor is closed")
        if self._pending >= self.capacity:
            raise PreprocessingQueueFullError("image preprocessing queue is full")
        self._pending += 1
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, preprocess_image, payload)

        def release(_future: asyncio.Future[np.ndarray]) -> None:
            self._pending = max(self._pending - 1, 0)

        future.add_done_callback(release)
        return future

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)


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
        preprocessor = _build_preprocessor(app_settings)
        runtime = InferenceRuntime(
            selected_backend,
            app_settings.scheduler,
            metrics=metrics,
        )
        app.state.runtime = runtime
        app.state.preprocessor = preprocessor
        try:
            await runtime.start()
            yield
        finally:
            try:
                await runtime.close()
            finally:
                await preprocessor.close()

    application = FastAPI(
        title="Adaptive Inference Gateway",
        version="0.1.0",
        description="Bounded image inference with direct or dynamic scheduling.",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    admission = PredictionAdmissionController(app_settings.server.max_in_flight_requests)
    application.state.prediction_admission = admission
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=(
            app_settings.server.max_upload_bytes + MULTIPART_OVERHEAD_ALLOWANCE_BYTES
        ),
    )
    application.add_middleware(
        PredictionAdmissionMiddleware,
        admission=admission,
        request_timeout_ms=app_settings.scheduler.request_timeout_ms,
    )
    # Starlette inserts newly added middleware at the outside. Metrics must be
    # added last so admission-generated 429 responses remain observable.
    application.add_middleware(PredictionMetricsMiddleware)

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
        started, deadline = _prediction_timing(request)
        runtime = _runtime(request)
        if not runtime.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="inference runtime is not ready",
            )

        request_id = uuid.uuid4().hex
        try:
            _raise_if_deadline_expired(deadline)
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
            _raise_if_deadline_expired(deadline)
            preprocessing_started = time.monotonic()
            upload_and_parse_ms = (preprocessing_started - started) * 1000.0
            preprocess_future = _preprocessor(request).submit(payload)
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RequestDeadlineExpiredError
            preprocess_timeout = asyncio.timeout(remaining_seconds)
            try:
                async with preprocess_timeout:
                    tensor = await asyncio.shield(preprocess_future)
            except TimeoutError as exc:
                if not preprocess_timeout.expired():
                    raise
                raise RequestDeadlineExpiredError from exc
            preprocessing_ms = (time.monotonic() - preprocessing_started) * 1000.0
            result = await runtime.predict(
                tensor,
                request_id=request_id,
                deadline_monotonic=deadline,
            )
        except PreprocessingQueueFullError as exc:
            runtime.stats.record_rejected()
            runtime.metrics.record_queue_rejection()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="image preprocessing capacity is full",
                headers={"Retry-After": "0"},
            ) from exc
        except RequestDeadlineExpiredError as exc:
            _record_timeout_once(request.scope, runtime)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="inference request exceeded its deadline",
            ) from exc
        except ImageTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
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
            # The scheduler owns the increment for this path. Mark the scope so
            # the outer absolute timeout cannot count the same deadline twice.
            _mark_timeout_recorded(request.scope)
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
        except HTTPException:
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
            server_processing_ms=(time.monotonic() - started) * 1000.0,
            upload_and_parse_ms=upload_and_parse_ms,
            preprocessing_ms=preprocessing_ms,
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


def _preprocessor(request: Request) -> BoundedPreprocessor:
    preprocessor = getattr(request.app.state, "preprocessor", None)
    if not isinstance(preprocessor, BoundedPreprocessor):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="image preprocessor has not started",
        )
    return preprocessor


def _runtime_from_scope(scope: Scope) -> InferenceRuntime | None:
    application = scope.get("app")
    state = getattr(application, "state", None)
    runtime = getattr(state, "runtime", None)
    return runtime if isinstance(runtime, InferenceRuntime) else None


def _build_preprocessor(settings: GatewaySettings) -> BoundedPreprocessor:
    cpu_count = os.cpu_count() or 1
    workers = min(settings.server.preprocessing_workers, cpu_count)
    capacity = settings.server.max_preprocessing_queue_size
    return BoundedPreprocessor(capacity=capacity, max_workers=workers)


def _is_prediction_request(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope.get("method") == "POST"
        and scope.get("path") == "/v1/predict"
    )


def _prediction_timing(request: Request) -> tuple[float, float]:
    scope_state = request.scope.get("state", {})
    arrival = scope_state.get(PREDICTION_ARRIVAL_STATE_KEY)
    deadline = scope_state.get(PREDICTION_DEADLINE_STATE_KEY)
    if (
        not isinstance(arrival, (int, float))
        or isinstance(arrival, bool)
        or not isinstance(deadline, (int, float))
        or isinstance(deadline, bool)
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="prediction admission state is unavailable",
        )
    return float(arrival), float(deadline)


def _raise_if_deadline_expired(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RequestDeadlineExpiredError


def _record_timeout_once(scope: Scope, runtime: InferenceRuntime | None) -> None:
    if runtime is None:
        return
    scope_state = scope.setdefault("state", {})
    if scope_state.get(PREDICTION_TIMEOUT_RECORDED_STATE_KEY) is True:
        return
    _mark_timeout_recorded(scope)
    runtime.stats.record_timeout()


def _mark_timeout_recorded(scope: Scope) -> None:
    scope.setdefault("state", {})[PREDICTION_TIMEOUT_RECORDED_STATE_KEY] = True


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _body_too_large_response(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={"detail": f"request body exceeds the {limit}-byte transport limit"},
    )


def _admission_rejected_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "prediction admission capacity is full"},
        headers={"Retry-After": "0"},
    )


def _deadline_exceeded_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": "inference request exceeded its deadline"},
    )


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


__all__ = [
    "ACCEPTED_MEDIA_TYPES",
    "BoundedPreprocessor",
    "PredictionAdmissionController",
    "PredictionAdmissionMiddleware",
    "PredictionMetricsMiddleware",
    "PreprocessingQueueFullError",
    "RequestBodyLimitMiddleware",
    "build_backend",
    "create_app",
]

"""Public HTTP response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: str


class ReadinessResponse(ApiModel):
    ready: bool
    backend_loaded: bool
    warmup_completed: bool
    scheduler_running: bool


class Prediction(ApiModel):
    index: int = Field(ge=0, le=999)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(ApiModel):
    request_id: str
    top_prediction_index: int = Field(ge=0, le=999)
    top_prediction_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    predictions: list[Prediction]
    backend: str
    device: str
    scheduler_mode: str
    server_processing_ms: float = Field(ge=0.0)
    upload_and_parse_ms: float = Field(ge=0.0)
    preprocessing_ms: float = Field(ge=0.0)
    queue_wait_ms: float = Field(ge=0.0)
    backend_inference_ms: float = Field(ge=0.0)
    realized_batch_size: int = Field(ge=1)


class ErrorResponse(ApiModel):
    detail: str

"""Validated YAML configuration for the inference gateway."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendName(StrEnum):
    """Supported execution backends."""

    TORCH = "torch"
    ONNX = "onnx"
    FAKE = "fake"


class DeviceName(StrEnum):
    """Portable device selections."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class SchedulerMode(StrEnum):
    """Request scheduling strategies."""

    DIRECT = "direct"
    DYNAMIC = "dynamic"


class StrictModel(BaseModel):
    """Base class that catches misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    architecture: str = "resnet18"
    backend: BackendName = BackendName.ONNX
    device: DeviceName = DeviceName.AUTO
    onnx_path: Path = Path("artifacts/resnet18.onnx")

    @field_validator("architecture")
    @classmethod
    def supported_architecture(cls, value: str) -> str:
        if value != "resnet18":
            raise ValueError("only the resnet18 architecture is supported")
        return value


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    top_k: int = Field(default=5, ge=1, le=1000)


class SchedulerConfig(StrictModel):
    mode: SchedulerMode = SchedulerMode.DYNAMIC
    max_batch_size: int = Field(default=16, ge=1)
    max_wait_ms: float = Field(default=2.0, ge=0.0)
    max_queue_size: int = Field(default=256, ge=1)
    request_timeout_ms: float = Field(default=5000.0, gt=0.0)
    inference_workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.mode is SchedulerMode.DYNAMIC and self.max_wait_ms >= self.request_timeout_ms:
            raise ValueError("dynamic max_wait_ms must be shorter than request_timeout_ms")
        return self


class ObservabilityConfig(StrictModel):
    prometheus: bool = True


class BenchmarkConfig(StrictModel):
    warmup_requests: int = Field(default=25, ge=0)
    requests_per_case: int = Field(default=200, ge=1)
    concurrency: list[int] = Field(default_factory=lambda: [1, 8, 32, 64], min_length=1)
    dynamic_batch_sizes: list[int] = Field(default_factory=lambda: [8, 16], min_length=1)
    dynamic_wait_ms: list[float] = Field(default_factory=lambda: [1.0, 3.0], min_length=1)
    synthetic_image_count: int = Field(default=8, ge=1)

    @field_validator("concurrency", "dynamic_batch_sizes")
    @classmethod
    def positive_unique_integers(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("values must be positive")
        if len(values) != len(set(values)):
            raise ValueError("values must be unique")
        return values

    @field_validator("dynamic_wait_ms")
    @classmethod
    def nonnegative_unique_waits(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("wait times cannot be negative")
        if len(values) != len(set(values)):
            raise ValueError("wait times must be unique")
        return values

    @property
    def dynamic_concurrency(self) -> list[int]:
        """Dynamic cases omit serial concurrency by design."""

        return [value for value in self.concurrency if value > 1]

    @property
    def primary_case_count(self) -> int:
        direct = 2 * len(self.concurrency)
        dynamic = 2 * len(self.dynamic_concurrency) * len(self.dynamic_batch_sizes) * len(
            self.dynamic_wait_ms
        )
        return direct + dynamic


class OutputConfig(StrictModel):
    directory: Path = Path("artifacts")


class GatewaySettings(BaseSettings):
    """Complete application settings, validated before startup."""

    model_config = SettingsConfigDict(
        env_prefix="AIG_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_combinations(self) -> Self:
        if self.model.backend is BackendName.ONNX and self.model.onnx_path.suffix != ".onnx":
            raise ValueError("model.onnx_path must end in .onnx")
        if self.benchmark.primary_case_count != 32:
            raise ValueError(
                "the primary benchmark matrix must contain exactly 32 cases "
                f"(configured {self.benchmark.primary_case_count})"
            )
        return self


def load_config(path: str | Path) -> GatewaySettings:
    """Load and validate a gateway YAML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return GatewaySettings.model_validate(raw)


def dump_config(settings: GatewaySettings, path: str | Path) -> Path:
    """Write validated settings to YAML for an isolated benchmark server."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(mode="json")
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path

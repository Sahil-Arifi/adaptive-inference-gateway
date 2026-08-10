"""Common inference backend contract and validation helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatBatch = npt.NDArray[np.float32]


class BackendError(RuntimeError):
    """Base class for backend configuration and execution failures."""


class BackendConfigurationError(BackendError):
    """Raised when a requested device or model configuration is unavailable."""


class BackendClosedError(BackendError):
    """Raised when inference is attempted after a backend has closed."""


class BackendExecutionError(BackendError):
    """Raised when a runtime returns an invalid result."""


def validate_batch(batch: np.ndarray) -> FloatBatch:
    """Validate a non-empty batched array and normalize it to contiguous float32."""

    if not isinstance(batch, np.ndarray):
        raise TypeError("Inference input must be a NumPy array.")
    if batch.ndim < 2:
        raise ValueError("Inference input must include batch and feature dimensions.")
    if batch.shape[0] < 1 or any(dimension < 1 for dimension in batch.shape[1:]):
        raise ValueError(f"Inference input dimensions must be positive; received {batch.shape}.")
    if not np.issubdtype(batch.dtype, np.number):
        raise TypeError("Inference input must contain numeric values.")
    return np.ascontiguousarray(batch, dtype=np.float32)


@runtime_checkable
class InferenceBackend(Protocol):
    """Synchronous execution contract used by both scheduler modes.

    Implementations are deliberately synchronous because PyTorch and ONNX
    Runtime calls block.  The scheduler is responsible for running them in its
    dedicated executor rather than on the HTTP event loop.
    """

    @property
    def name(self) -> str:
        """Stable backend identifier used in responses and metrics."""

        ...

    @property
    def device(self) -> str:
        """Resolved execution device, never the unresolved value ``auto``."""

        ...

    def predict_logits(self, batch: np.ndarray) -> FloatBatch:
        """Run one physical inference call for a logical batch."""

        ...

    def warmup(self) -> None:
        """Execute any runtime initialization needed before readiness."""

        ...

    def close(self) -> None:
        """Release resources and reject subsequent inference calls."""

        ...


__all__ = [
    "BackendClosedError",
    "BackendConfigurationError",
    "BackendError",
    "BackendExecutionError",
    "FloatBatch",
    "InferenceBackend",
    "validate_batch",
]

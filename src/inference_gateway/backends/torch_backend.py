"""PyTorch inference backend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from inference_gateway.backends.base import (
    BackendClosedError,
    BackendConfigurationError,
    BackendExecutionError,
    FloatBatch,
    validate_batch,
)


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve ``cpu``, ``cuda``/``cuda:N``, or ``auto`` to a usable device."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise BackendConfigurationError(
                "CUDA was requested for PyTorch, but torch.cuda.is_available() is false."
            )
        try:
            return torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise BackendConfigurationError(f"Invalid PyTorch device {requested!r}.") from exc
    raise BackendConfigurationError("PyTorch device must be 'cpu', 'cuda', 'cuda:N', or 'auto'.")


class TorchBackend:
    """Run a torchvision ResNet18 or an injected local test model."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        device: str = "auto",
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        input_shape: Sequence[int] = (3, 224, 224),
        warmup_batch_size: int = 1,
    ) -> None:
        if not input_shape or any(dimension < 1 for dimension in input_shape):
            raise ValueError("input_shape must contain positive dimensions.")
        if warmup_batch_size < 1:
            raise ValueError("warmup_batch_size must be positive.")

        self._torch_device = resolve_torch_device(device)
        self._model = model if model is not None else resnet18(weights=weights)
        self._model.to(self._torch_device)
        self._model.eval()
        self._input_shape = tuple(int(dimension) for dimension in input_shape)
        self._warmup_batch_size = warmup_batch_size
        self._closed = False

    @property
    def name(self) -> str:
        return "torch"

    @property
    def device(self) -> str:
        return str(self._torch_device)

    @property
    def model(self) -> nn.Module:
        return self._model

    def predict_logits(self, batch: np.ndarray) -> FloatBatch:
        if self._closed:
            raise BackendClosedError("The PyTorch backend is closed.")
        values = validate_batch(batch)
        inputs = torch.from_numpy(values).to(self._torch_device)
        with torch.inference_mode():
            output = self._model(inputs)
        if not isinstance(output, torch.Tensor):
            raise BackendExecutionError("The PyTorch model did not return a tensor.")
        if output.ndim < 2 or output.shape[0] != values.shape[0]:
            raise BackendExecutionError(
                "The PyTorch model returned an invalid batch shape: "
                f"input {values.shape}, output {tuple(output.shape)}."
            )
        array = output.detach().to(device="cpu", dtype=torch.float32).numpy()
        return np.ascontiguousarray(array, dtype=np.float32)

    def warmup(self) -> None:
        if self._closed:
            raise BackendClosedError("The PyTorch backend is closed.")
        warmup = np.zeros(
            (self._warmup_batch_size, *self._input_shape),
            dtype=np.float32,
        )
        self.predict_logits(warmup)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._torch_device.type == "cuda":
            torch.cuda.synchronize(self._torch_device)


__all__ = ["TorchBackend", "resolve_torch_device"]

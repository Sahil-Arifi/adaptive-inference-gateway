"""ONNX Runtime inference backend with CPU and optional CUDA execution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from inference_gateway.backends.base import (
    BackendClosedError,
    BackendConfigurationError,
    BackendExecutionError,
    FloatBatch,
    validate_batch,
)


def resolve_onnx_providers(requested: str) -> tuple[list[str], str]:
    """Return ordered providers and the corresponding resolved device name."""

    normalized = requested.strip().lower()
    available = set(ort.get_available_providers())
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"

    if normalized == "auto":
        if cuda in available:
            providers = [cuda]
            if cpu in available:
                providers.append(cpu)
            return providers, "cuda"
        if cpu in available:
            return [cpu], "cpu"
        raise BackendConfigurationError("ONNX Runtime has no supported CPU or CUDA provider.")
    if normalized == "cpu":
        if cpu not in available:
            raise BackendConfigurationError("CPUExecutionProvider is not installed.")
        return [cpu], "cpu"
    if normalized == "cuda":
        if cuda not in available:
            raise BackendConfigurationError(
                "CUDA was requested for ONNX Runtime, but CUDAExecutionProvider is not installed."
            )
        providers = [cuda]
        if cpu in available:
            providers.append(cpu)
        return providers, "cuda"
    raise BackendConfigurationError("ONNX Runtime device must be 'cpu', 'cuda', or 'auto'.")


class OnnxBackend:
    """Execute a single-output image classifier with ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        input_shape: Sequence[int] | None = None,
        warmup_batch_size: int = 1,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {path}")
        if warmup_batch_size < 1:
            raise ValueError("warmup_batch_size must be positive.")

        providers, resolved_device = resolve_onnx_providers(device)
        self._session: ort.InferenceSession | None = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=providers,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise BackendConfigurationError(
                f"Expected exactly one ONNX input, found {len(inputs)}."
            )
        if not outputs:
            raise BackendConfigurationError("The ONNX model does not define an output.")

        self._model_path = path
        self._device = resolved_device
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = self._resolve_input_shape(inputs[0].shape, input_shape)
        self._warmup_batch_size = warmup_batch_size

    @staticmethod
    def _resolve_input_shape(
        metadata_shape: list[int | str | None],
        configured_shape: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if configured_shape is not None:
            shape = tuple(int(dimension) for dimension in configured_shape)
            if not shape or any(dimension < 1 for dimension in shape):
                raise ValueError("input_shape must contain positive dimensions.")
            return shape

        tail = metadata_shape[1:]
        if not tail or any(not isinstance(dimension, int) or dimension < 1 for dimension in tail):
            raise BackendConfigurationError(
                "The ONNX input has dynamic non-batch dimensions; provide input_shape explicitly."
            )
        return tuple(cast(int, dimension) for dimension in tail)

    @property
    def name(self) -> str:
        return "onnx"

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def input_name(self) -> str:
        return cast(str, self._input_name)

    @property
    def output_name(self) -> str:
        return cast(str, self._output_name)

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            raise BackendClosedError("The ONNX backend is closed.")
        return self._session

    def predict_logits(self, batch: np.ndarray) -> FloatBatch:
        values = validate_batch(batch)
        session = self.session
        raw_outputs = session.run([self._output_name], {self._input_name: values})
        if len(raw_outputs) != 1:
            raise BackendExecutionError(
                f"ONNX Runtime returned {len(raw_outputs)} outputs; expected one."
            )
        output = np.asarray(raw_outputs[0], dtype=np.float32)
        if output.ndim < 2 or output.shape[0] != values.shape[0]:
            raise BackendExecutionError(
                "ONNX Runtime returned an invalid batch shape: "
                f"input {values.shape}, output {output.shape}."
            )
        return np.ascontiguousarray(output, dtype=np.float32)

    def warmup(self) -> None:
        if self._session is None:
            raise BackendClosedError("The ONNX backend is closed.")
        warmup = np.zeros(
            (self._warmup_batch_size, *self._input_shape),
            dtype=np.float32,
        )
        self.predict_logits(warmup)

    def close(self) -> None:
        self._session = None


__all__ = ["OnnxBackend", "resolve_onnx_providers"]

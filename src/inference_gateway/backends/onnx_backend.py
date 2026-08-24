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

PRODUCTION_INPUT_SHAPE = (3, 224, 224)
PRODUCTION_OUTPUT_SIZE = 1000


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
        return [cuda], "cuda"
    raise BackendConfigurationError("ONNX Runtime device must be 'cpu', 'cuda', or 'auto'.")


def resolve_active_onnx_device(
    session: ort.InferenceSession,
    *,
    expected_device: str,
) -> str:
    """Resolve the session's highest-priority active provider and reject fallback."""

    providers = session.get_providers()
    active_provider = providers[0] if providers else None
    if active_provider == "CUDAExecutionProvider":
        active_device = "cuda"
    elif active_provider == "CPUExecutionProvider":
        active_device = "cpu"
    else:
        raise BackendConfigurationError(
            "ONNX Runtime did not activate a supported CPU or CUDA execution provider; "
            f"found {providers}."
        )
    if active_device != expected_device:
        raise BackendConfigurationError(
            "ONNX Runtime silently fell back from the requested "
            f"{expected_device!r} device to active provider {active_provider!r}."
        )
    return active_device


class OnnxBackend:
    """Execute a single-output image classifier with ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        input_shape: Sequence[int] | None = None,
        expected_output_size: int = PRODUCTION_OUTPUT_SIZE,
        warmup_batch_size: int = 1,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {path}")
        if warmup_batch_size < 1:
            raise ValueError("warmup_batch_size must be positive.")
        if expected_output_size < 1:
            raise ValueError("expected_output_size must be positive.")

        providers, resolved_device = resolve_onnx_providers(device)
        self._session: ort.InferenceSession | None = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=providers,
        )
        active_device = resolve_active_onnx_device(
            self._session,
            expected_device=resolved_device,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise BackendConfigurationError(
                f"Expected exactly one ONNX input, found {len(inputs)}."
            )
        if len(outputs) != 1:
            raise BackendConfigurationError(
                f"Expected exactly one ONNX output, found {len(outputs)}."
            )

        self._model_path = path
        self._device = active_device
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = self._validate_input_contract(inputs[0], input_shape)
        self._expected_output_size = expected_output_size
        self._validate_output_contract(outputs[0], expected_output_size)
        self._warmup_batch_size = warmup_batch_size

    @staticmethod
    def _require_float_tensor(metadata: ort.NodeArg, role: str) -> None:
        element_type = cast(str, metadata.type)
        if element_type != "tensor(float)":
            raise BackendConfigurationError(
                f"The ONNX {role} must be tensor(float), found {element_type!r}."
            )

    @staticmethod
    def _require_dynamic_batch(metadata: ort.NodeArg, role: str) -> None:
        metadata_shape = metadata.shape
        if not metadata_shape:
            raise BackendConfigurationError(f"The ONNX {role} must include a batch dimension.")
        batch_dimension = metadata_shape[0]
        if batch_dimension is not None and not (
            isinstance(batch_dimension, str) and bool(batch_dimension)
        ):
            raise BackendConfigurationError(
                f"The ONNX {role} batch dimension must be dynamic, found {batch_dimension!r}."
            )

    @classmethod
    def _validate_input_contract(
        cls,
        metadata: ort.NodeArg,
        configured_shape: Sequence[int] | None,
    ) -> tuple[int, ...]:
        cls._require_float_tensor(metadata, "input")
        cls._require_dynamic_batch(metadata, "input")
        expected = (
            tuple(int(dimension) for dimension in configured_shape)
            if configured_shape is not None
            else PRODUCTION_INPUT_SHAPE
        )
        if not expected or any(dimension < 1 for dimension in expected):
            raise ValueError("input_shape must contain positive dimensions.")
        metadata_shape = metadata.shape
        tail = metadata_shape[1:]
        if len(tail) != len(expected) or any(
            not isinstance(dimension, int) or dimension < 1 for dimension in tail
        ):
            raise BackendConfigurationError(
                "The ONNX input must have static, positive non-batch dimensions; "
                f"found {metadata_shape}."
            )
        actual = tuple(cast(int, dimension) for dimension in tail)
        if actual != expected:
            raise BackendConfigurationError(
                f"The ONNX input tail must be {expected}, found {actual}."
            )
        return expected

    @classmethod
    def _validate_output_contract(
        cls,
        metadata: ort.NodeArg,
        expected_output_size: int,
    ) -> None:
        cls._require_float_tensor(metadata, "output")
        cls._require_dynamic_batch(metadata, "output")
        metadata_shape = metadata.shape
        tail = metadata_shape[1:]
        if (
            len(metadata_shape) != 2
            or len(tail) != 1
            or not isinstance(tail[0], int)
            or tail[0] != expected_output_size
        ):
            raise BackendConfigurationError(
                "The ONNX classifier output must have shape "
                f"[batch, {expected_output_size}], found {metadata_shape}."
            )

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
    def expected_output_size(self) -> int:
        return self._expected_output_size

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            raise BackendClosedError("The ONNX backend is closed.")
        return self._session

    def predict_logits(self, batch: np.ndarray) -> FloatBatch:
        values = validate_batch(batch)
        if tuple(values.shape[1:]) != self._input_shape:
            raise BackendExecutionError(
                f"ONNX input tail must be {self._input_shape}, found {values.shape[1:]}."
            )
        session = self.session
        raw_outputs = session.run([self._output_name], {self._input_name: values})
        if len(raw_outputs) != 1:
            raise BackendExecutionError(
                f"ONNX Runtime returned {len(raw_outputs)} outputs; expected one."
            )
        output = np.asarray(raw_outputs[0], dtype=np.float32)
        if output.shape != (values.shape[0], self._expected_output_size):
            raise BackendExecutionError(
                "ONNX Runtime returned an invalid classifier output shape: "
                f"expected {(values.shape[0], self._expected_output_size)}, "
                f"found {output.shape}."
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


__all__ = [
    "PRODUCTION_INPUT_SHAPE",
    "PRODUCTION_OUTPUT_SIZE",
    "OnnxBackend",
    "resolve_active_onnx_device",
    "resolve_onnx_providers",
]

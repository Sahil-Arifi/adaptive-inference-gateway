"""Deterministic numerical parity checks for PyTorch and ONNX Runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from inference_gateway.backends.base import BackendConfigurationError
from inference_gateway.backends.onnx_backend import (
    resolve_active_onnx_device,
    resolve_onnx_providers,
)
from inference_gateway.backends.torch_backend import resolve_torch_device

PARITY_SCHEMA_VERSION = 1
PRODUCTION_PARITY_ATOL = 1e-5
PRODUCTION_PARITY_BATCH_SIZES = (1, 4, 16)
PRODUCTION_PARITY_DEVICE = "cpu"
PRODUCTION_PARITY_INPUT_SHAPE = (3, 224, 224)
PRODUCTION_PARITY_MODEL_NAME = "resnet18"
PRODUCTION_PARITY_OUTPUT_SIZE = 1000
PRODUCTION_PARITY_RTOL = 1e-4


class ParityVerificationError(RuntimeError):
    """Raised when parity fails or a parity artifact cannot be trusted."""


@dataclass(frozen=True, slots=True)
class BatchParityResult:
    """Numerical comparison for one batch size."""

    batch_size: int
    logit_count: int
    max_abs_difference: float
    mean_abs_difference: float
    top1_agreement: float
    allclose: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "logit_count": self.logit_count,
            "max_abs_difference": self.max_abs_difference,
            "mean_abs_difference": self.mean_abs_difference,
            "top1_agreement": self.top1_agreement,
            "allclose": self.allclose,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BatchParityResult:
        result = cls(
            batch_size=_positive_int(data, "batch_size"),
            logit_count=_positive_int(data, "logit_count"),
            max_abs_difference=_nonnegative_float(data, "max_abs_difference"),
            mean_abs_difference=_nonnegative_float(data, "mean_abs_difference"),
            top1_agreement=_fraction(data, "top1_agreement"),
            allclose=_boolean(data, "allclose"),
            passed=_boolean(data, "passed"),
        )
        if result.logit_count % result.batch_size:
            raise ValueError("logit_count must be divisible by batch_size.")
        expected_pass = result.allclose and result.top1_agreement == 1.0
        if result.passed != expected_pass:
            raise ValueError("A per-batch passed flag is inconsistent with its metrics.")
        return result


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Artifact-bound PyTorch/ONNX parity evidence."""

    schema_version: int
    model_name: str
    onnx_path: str
    onnx_sha256: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    rtol: float
    atol: float
    seed: int
    device: str
    per_batch: tuple[BatchParityResult, ...]
    max_abs_difference: float
    mean_abs_difference: float
    top1_agreement: float
    passed: bool

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(result.batch_size for result in self.per_batch)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "onnx_path": self.onnx_path,
            "onnx_sha256": self.onnx_sha256,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": list(self.input_shape),
            "batch_sizes": list(self.batch_sizes),
            "rtol": self.rtol,
            "atol": self.atol,
            "seed": self.seed,
            "device": self.device,
            "per_batch": [result.to_dict() for result in self.per_batch],
            "max_abs_difference": self.max_abs_difference,
            "mean_abs_difference": self.mean_abs_difference,
            "top1_agreement": self.top1_agreement,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ParityReport:
        schema_version = _positive_int(data, "schema_version")
        if schema_version != PARITY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported parity schema version {schema_version}.")
        shape_value = data.get("input_shape")
        if not isinstance(shape_value, list) or not shape_value:
            raise ValueError("input_shape must be a non-empty list.")
        input_shape = tuple(_plain_positive_int(item, "input_shape") for item in shape_value)
        batches_value = data.get("per_batch")
        if not isinstance(batches_value, list) or not batches_value:
            raise ValueError("per_batch must be a non-empty list.")
        batches: list[BatchParityResult] = []
        for item in batches_value:
            if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
                raise ValueError("Each per_batch entry must be an object.")
            batches.append(BatchParityResult.from_dict(cast(dict[str, object], item)))
        declared_sizes = data.get("batch_sizes")
        expected_sizes = [batch.batch_size for batch in batches]
        if declared_sizes != expected_sizes:
            raise ValueError("batch_sizes does not match per_batch entries.")

        sha256 = _string(data, "onnx_sha256")
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("onnx_sha256 must be a lowercase SHA-256 digest.")
        seed_value = data.get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise ValueError("seed must be an integer.")

        report = cls(
            schema_version=schema_version,
            model_name=_string(data, "model_name"),
            onnx_path=_string(data, "onnx_path"),
            onnx_sha256=sha256,
            input_name=_string(data, "input_name"),
            output_name=_string(data, "output_name"),
            input_shape=input_shape,
            rtol=_nonnegative_float(data, "rtol"),
            atol=_nonnegative_float(data, "atol"),
            seed=seed_value,
            device=_string(data, "device"),
            per_batch=tuple(batches),
            max_abs_difference=_nonnegative_float(data, "max_abs_difference"),
            mean_abs_difference=_nonnegative_float(data, "mean_abs_difference"),
            top1_agreement=_fraction(data, "top1_agreement"),
            passed=_boolean(data, "passed"),
        )
        expected_pass = all(batch.passed for batch in report.per_batch)
        expected_max = max(batch.max_abs_difference for batch in report.per_batch)
        total_logits = sum(batch.logit_count for batch in report.per_batch)
        expected_mean = (
            sum(batch.mean_abs_difference * batch.logit_count for batch in report.per_batch)
            / total_logits
        )
        total_predictions = sum(batch.batch_size for batch in report.per_batch)
        expected_top1 = (
            sum(batch.top1_agreement * batch.batch_size for batch in report.per_batch)
            / total_predictions
        )
        if report.passed != expected_pass:
            raise ValueError("The aggregate passed flag is inconsistent with per_batch entries.")
        if not np.isclose(report.max_abs_difference, expected_max, rtol=1e-12, atol=1e-15):
            raise ValueError("The aggregate maximum difference is inconsistent.")
        if not np.isclose(report.mean_abs_difference, expected_mean, rtol=1e-12, atol=1e-15):
            raise ValueError("The aggregate mean difference is inconsistent.")
        if not np.isclose(report.top1_agreement, expected_top1, rtol=1e-12, atol=1e-15):
            raise ValueError("The aggregate top-1 agreement is inconsistent.")
        return report


def _string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _boolean(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _plain_positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must contain positive integers.")
    return value


def _positive_int(data: Mapping[str, object], key: str) -> int:
    return _plain_positive_int(data.get(key), key)


def _nonnegative_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric.")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{key} must be finite and non-negative.")
    return result


def _fraction(data: Mapping[str, object], key: str) -> float:
    result = _nonnegative_float(data, key)
    if result > 1:
        raise ValueError(f"{key} must be between zero and one.")
    return result


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading the whole model artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_input_shape(input_metadata: ort.NodeArg) -> tuple[int, ...]:
    tail = input_metadata.shape[1:]
    if not tail or any(not isinstance(dimension, int) or dimension < 1 for dimension in tail):
        raise ParityVerificationError(
            "Cannot infer dynamic non-batch dimensions; provide input_shape explicitly."
        )
    return tuple(int(dimension) for dimension in tail)


def verify_parity(
    model: nn.Module,
    onnx_path: str | Path,
    *,
    batch_sizes: Sequence[int] = (1, 4, 16),
    input_shape: Sequence[int] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    seed: int = 2027,
    device: str = "cpu",
    model_name: str | None = None,
) -> ParityReport:
    """Compare both runtimes on the exact same deterministic input tensors."""

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")
    normalized_batches = tuple(int(size) for size in batch_sizes)
    if not normalized_batches or any(size < 1 for size in normalized_batches):
        raise ValueError("batch_sizes must contain positive integers.")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol cannot be negative.")

    try:
        model_proto = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model_proto)
    except Exception as exc:
        raise ParityVerificationError(f"ONNX checker rejected {path}.") from exc

    requested_device = device.strip().lower()
    if requested_device == "auto":
        common_device = (
            "cuda"
            if torch.cuda.is_available()
            and "CUDAExecutionProvider" in ort.get_available_providers()
            else "cpu"
        )
    else:
        common_device = requested_device
    providers, resolved_onnx_device = resolve_onnx_providers(common_device)
    torch_device = resolve_torch_device(common_device)
    if torch_device.type != resolved_onnx_device:
        raise ParityVerificationError(
            "PyTorch and ONNX Runtime resolved to different device types: "
            f"{torch_device.type} and {resolved_onnx_device}."
        )
    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:
        raise ParityVerificationError(f"ONNX Runtime could not load {path}.") from exc
    try:
        active_onnx_device = resolve_active_onnx_device(
            session,
            expected_device=resolved_onnx_device,
        )
    except BackendConfigurationError as exc:
        raise ParityVerificationError(
            "ONNX Runtime did not activate the device selected for parity."
        ) from exc
    if torch_device.type != active_onnx_device:
        raise ParityVerificationError(
            "PyTorch and the active ONNX Runtime provider use different device types: "
            f"{torch_device.type} and {active_onnx_device}."
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise ParityVerificationError("Parity requires one ONNX input and at least one output.")
    resolved_shape = (
        tuple(int(dimension) for dimension in input_shape)
        if input_shape is not None
        else _infer_input_shape(inputs[0])
    )
    if not resolved_shape or any(dimension < 1 for dimension in resolved_shape):
        raise ValueError("input_shape must contain positive dimensions.")

    model.to(torch_device)
    model.eval()
    rng = np.random.default_rng(seed)
    results: list[BatchParityResult] = []
    absolute_sum = 0.0
    total_logits = 0
    total_top1_matches = 0
    total_predictions = 0

    for batch_size in normalized_batches:
        values = rng.standard_normal((batch_size, *resolved_shape), dtype=np.float32)
        with torch.inference_mode():
            torch_raw = model(torch.from_numpy(values).to(torch_device))
        if not isinstance(torch_raw, torch.Tensor):
            raise ParityVerificationError("The PyTorch model did not return a tensor.")
        torch_output = torch_raw.detach().to(device="cpu", dtype=torch.float32).numpy()
        onnx_raw = session.run([outputs[0].name], {inputs[0].name: values})[0]
        onnx_output = np.asarray(onnx_raw, dtype=np.float32)
        if torch_output.shape != onnx_output.shape:
            raise ParityVerificationError(
                f"Batch {batch_size} shape mismatch: PyTorch {torch_output.shape}, "
                f"ONNX {onnx_output.shape}."
            )
        if torch_output.ndim != 2 or torch_output.shape[0] != batch_size:
            raise ParityVerificationError(
                f"Expected [batch, classes] logits, received {torch_output.shape}."
            )
        if not np.isfinite(torch_output).all() or not np.isfinite(onnx_output).all():
            raise ParityVerificationError(f"Batch {batch_size} produced non-finite logits.")

        difference = np.abs(
            torch_output.astype(np.float64) - onnx_output.astype(np.float64)
        )
        max_difference = float(difference.max(initial=0.0))
        mean_difference = float(difference.mean())
        allclose = bool(np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol))
        torch_top1 = np.argmax(torch_output, axis=1)
        onnx_top1 = np.argmax(onnx_output, axis=1)
        matches = int(np.count_nonzero(torch_top1 == onnx_top1))
        agreement = matches / batch_size
        passed = allclose and matches == batch_size
        logit_count = int(difference.size)
        results.append(
            BatchParityResult(
                batch_size=batch_size,
                logit_count=logit_count,
                max_abs_difference=max_difference,
                mean_abs_difference=mean_difference,
                top1_agreement=agreement,
                allclose=allclose,
                passed=passed,
            )
        )
        absolute_sum += float(difference.sum(dtype=np.float64))
        total_logits += logit_count
        total_top1_matches += matches
        total_predictions += batch_size

    resolved_model_name = model_name if model_name is not None else type(model).__name__
    if not isinstance(resolved_model_name, str) or not resolved_model_name:
        raise ValueError("model_name must be a non-empty string.")

    return ParityReport(
        schema_version=PARITY_SCHEMA_VERSION,
        model_name=resolved_model_name,
        onnx_path=str(path),
        onnx_sha256=sha256_file(path),
        input_name=inputs[0].name,
        output_name=outputs[0].name,
        input_shape=resolved_shape,
        rtol=float(rtol),
        atol=float(atol),
        seed=seed,
        device=active_onnx_device,
        per_batch=tuple(results),
        max_abs_difference=max(result.max_abs_difference for result in results),
        mean_abs_difference=absolute_sum / total_logits,
        top1_agreement=total_top1_matches / total_predictions,
        passed=all(result.passed for result in results),
    )


def write_parity_report(report: ParityReport, destination: str | Path) -> Path:
    """Serialize a parity report with full floating-point precision."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_parity_report(source: str | Path) -> ParityReport:
    """Load and strictly validate a parity artifact."""

    path = Path(source)
    if not path.is_file():
        raise ParityVerificationError(f"Parity report does not exist: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("The parity report root must be an object.")
        return ParityReport.from_dict(cast(dict[str, object], raw))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise ParityVerificationError(f"Invalid parity report: {path}") from exc


def validate_production_parity_policy(
    report: ParityReport,
    *,
    expected_device: str = PRODUCTION_PARITY_DEVICE,
) -> None:
    """Validate production benchmark parity evidence without reading external files."""

    policy_errors: list[str] = []
    if not report.passed or not all(result.passed for result in report.per_batch):
        policy_errors.append("the report records a failed comparison")
    if report.model_name != PRODUCTION_PARITY_MODEL_NAME:
        policy_errors.append(
            f"model identity must be {PRODUCTION_PARITY_MODEL_NAME!r}, "
            f"found {report.model_name!r}"
        )
    if report.input_shape != PRODUCTION_PARITY_INPUT_SHAPE:
        policy_errors.append(
            f"input shape must be {PRODUCTION_PARITY_INPUT_SHAPE}, "
            f"found {report.input_shape}"
        )
    if report.batch_sizes != PRODUCTION_PARITY_BATCH_SIZES:
        policy_errors.append(
            f"batch sizes must be exactly {PRODUCTION_PARITY_BATCH_SIZES}, "
            f"found {report.batch_sizes}"
        )
    if report.rtol != PRODUCTION_PARITY_RTOL:
        policy_errors.append(
            f"rtol must be {PRODUCTION_PARITY_RTOL}, found {report.rtol}"
        )
    if report.atol != PRODUCTION_PARITY_ATOL:
        policy_errors.append(
            f"atol must be {PRODUCTION_PARITY_ATOL}, found {report.atol}"
        )
    if report.device != expected_device:
        policy_errors.append(
            f"device must be {expected_device!r} for the configured benchmark, "
            f"found {report.device!r}"
        )
    invalid_output_batches = tuple(
        result.batch_size
        for result in report.per_batch
        if result.logit_count != result.batch_size * PRODUCTION_PARITY_OUTPUT_SIZE
    )
    if invalid_output_batches:
        policy_errors.append(
            f"classifier output must contain {PRODUCTION_PARITY_OUTPUT_SIZE} logits per image"
        )
    if policy_errors:
        raise ParityVerificationError(
            "The parity report does not satisfy the production benchmark policy: "
            + "; ".join(policy_errors)
        )


def require_passing_parity(
    parity_path: str | Path,
    onnx_path: str | Path,
    *,
    expected_device: str = PRODUCTION_PARITY_DEVICE,
) -> ParityReport:
    """Reject missing, failed, malformed, or stale parity evidence."""

    model_path = Path(onnx_path)
    if not model_path.is_file():
        raise ParityVerificationError(f"ONNX model does not exist: {model_path}")
    report = load_parity_report(parity_path)
    actual_sha256 = sha256_file(model_path)
    if report.onnx_sha256 != actual_sha256:
        raise ParityVerificationError(
            "The parity report does not match the current ONNX artifact SHA-256."
        )
    validate_production_parity_policy(report, expected_device=expected_device)
    return report


def verify_and_write_parity(
    model: nn.Module,
    onnx_path: str | Path,
    destination: str | Path,
    *,
    batch_sizes: Sequence[int] = (1, 4, 16),
    input_shape: Sequence[int] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    seed: int = 2027,
    device: str = "cpu",
    model_name: str | None = None,
) -> ParityReport:
    """Run parity, write its evidence, and fail closed when it does not pass."""

    report = verify_parity(
        model,
        onnx_path,
        batch_sizes=batch_sizes,
        input_shape=input_shape,
        rtol=rtol,
        atol=atol,
        seed=seed,
        device=device,
        model_name=model_name,
    )
    write_parity_report(report, destination)
    if not report.passed:
        raise ParityVerificationError("PyTorch/ONNX parity verification failed.")
    return report


# Explicit semantic alias used by CLI code.
compare_pytorch_onnx = verify_parity


__all__ = [
    "PARITY_SCHEMA_VERSION",
    "PRODUCTION_PARITY_ATOL",
    "PRODUCTION_PARITY_BATCH_SIZES",
    "PRODUCTION_PARITY_DEVICE",
    "PRODUCTION_PARITY_INPUT_SHAPE",
    "PRODUCTION_PARITY_MODEL_NAME",
    "PRODUCTION_PARITY_OUTPUT_SIZE",
    "PRODUCTION_PARITY_RTOL",
    "BatchParityResult",
    "ParityReport",
    "ParityVerificationError",
    "compare_pytorch_onnx",
    "load_parity_report",
    "require_passing_parity",
    "sha256_file",
    "validate_production_parity_policy",
    "verify_and_write_parity",
    "verify_parity",
    "write_parity_report",
]

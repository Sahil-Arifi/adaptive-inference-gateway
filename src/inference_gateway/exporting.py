"""Modern PyTorch-to-ONNX export and runtime validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class OnnxExportError(RuntimeError):
    """Raised when an exported artifact is invalid or not genuinely dynamic."""


@dataclass(frozen=True, slots=True)
class OnnxBatchValidation:
    """Observed output from one real ONNX Runtime execution."""

    batch_size: int
    output_shape: tuple[int, ...]
    all_finite: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_shape"] = list(self.output_shape)
        return data


@dataclass(frozen=True, slots=True)
class OnnxValidationReport:
    """Checker and ONNX Runtime evidence for one exported model."""

    onnx_path: str
    input_name: str
    output_name: str
    dynamic_input_batch: bool
    dynamic_output_batch: bool
    providers: tuple[str, ...]
    batches: tuple[OnnxBatchValidation, ...]

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(result.batch_size for result in self.batches)

    @property
    def passed(self) -> bool:
        return (
            self.dynamic_input_batch
            and self.dynamic_output_batch
            and bool(self.batches)
            and all(result.all_finite for result in self.batches)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "onnx_path": self.onnx_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "dynamic_input_batch": self.dynamic_input_batch,
            "dynamic_output_batch": self.dynamic_output_batch,
            "providers": list(self.providers),
            "batch_sizes": list(self.batch_sizes),
            "batches": [batch.to_dict() for batch in self.batches],
            "passed": self.passed,
        }


def _model_device(model: nn.Module) -> torch.device:
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None:
        return first_parameter.device
    first_buffer = next(model.buffers(), None)
    return first_buffer.device if first_buffer is not None else torch.device("cpu")


def _is_dynamic_batch(value_info: onnx.ValueInfoProto) -> bool:
    dimensions = value_info.type.tensor_type.shape.dim
    if not dimensions:
        return False
    return dimensions[0].WhichOneof("value") == "dim_param" and bool(dimensions[0].dim_param)


def export_model_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    sample_input: torch.Tensor | None = None,
    *,
    input_shape: Sequence[int] = (3, 224, 224),
    input_name: str = "images",
    output_name: str = "logits",
    opset_version: int = 18,
) -> Path:
    """Export *model* with the torch.export-based ONNX exporter.

    The batch dimension uses ``torch.export.Dim`` through ``dynamic_shapes``;
    this is the modern dynamo exporter path rather than the legacy tracing-only
    ``dynamic_axes`` mechanism.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sample_input is None:
        if not input_shape or any(dimension < 1 for dimension in input_shape):
            raise ValueError("input_shape must contain positive dimensions.")
        sample_input = torch.zeros((1, *input_shape), dtype=torch.float32)
    if sample_input.ndim < 2 or sample_input.shape[0] < 1:
        raise ValueError("sample_input must include a non-empty batch dimension.")

    model.eval()
    model_device = _model_device(model)
    example = sample_input.detach().to(device=model_device, dtype=torch.float32)
    dynamic_batch = torch.export.Dim("batch", min=1)

    try:
        torch.onnx.export(
            model,
            (example,),
            str(destination),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=({0: dynamic_batch},),
            external_data=False,
            verbose=False,
        )
    except Exception as exc:
        raise OnnxExportError(f"Failed to export ONNX model to {destination}.") from exc

    if not destination.is_file():
        raise OnnxExportError(f"The ONNX exporter did not create {destination}.")
    return destination


def _infer_input_shape(input_metadata: ort.NodeArg) -> tuple[int, ...]:
    dimensions = input_metadata.shape[1:]
    if not dimensions or any(
        not isinstance(dimension, int) or dimension < 1 for dimension in dimensions
    ):
        raise OnnxExportError(
            "Cannot infer dynamic non-batch input dimensions; provide input_shape explicitly."
        )
    return tuple(int(dimension) for dimension in dimensions)


def validate_onnx_model(
    onnx_path: str | Path,
    *,
    batch_sizes: Sequence[int] = (1, 4, 16),
    input_shape: Sequence[int] | None = None,
    seed: int = 1729,
) -> OnnxValidationReport:
    """Run ONNX checker and real CPU inference for each requested batch size."""

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")
    normalized_batches = tuple(int(size) for size in batch_sizes)
    if not normalized_batches or any(size < 1 for size in normalized_batches):
        raise ValueError("batch_sizes must contain positive integers.")

    try:
        model_proto = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model_proto)
    except Exception as exc:
        raise OnnxExportError(f"ONNX checker rejected {path}.") from exc
    if len(model_proto.graph.input) != 1:
        raise OnnxExportError(
            f"Expected one graph input, found {len(model_proto.graph.input)}."
        )
    if not model_proto.graph.output:
        raise OnnxExportError("The ONNX graph does not define an output.")

    dynamic_input = _is_dynamic_batch(model_proto.graph.input[0])
    dynamic_output = _is_dynamic_batch(model_proto.graph.output[0])
    if not dynamic_input or not dynamic_output:
        raise OnnxExportError(
            "The exported ONNX graph does not have dynamic input and output batch axes."
        )

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise OnnxExportError(f"ONNX Runtime could not load {path}.") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise OnnxExportError("ONNX Runtime did not expose one input and at least one output.")
    resolved_shape = (
        tuple(int(dimension) for dimension in input_shape)
        if input_shape is not None
        else _infer_input_shape(inputs[0])
    )
    if not resolved_shape or any(dimension < 1 for dimension in resolved_shape):
        raise ValueError("input_shape must contain positive dimensions.")

    rng = np.random.default_rng(seed)
    results: list[OnnxBatchValidation] = []
    for batch_size in normalized_batches:
        values = rng.standard_normal((batch_size, *resolved_shape), dtype=np.float32)
        try:
            raw_output: list[Any] = session.run([outputs[0].name], {inputs[0].name: values})
        except Exception as exc:
            raise OnnxExportError(
                f"ONNX Runtime inference failed for batch size {batch_size}."
            ) from exc
        output = np.asarray(raw_output[0])
        if output.ndim < 1 or output.shape[0] != batch_size:
            raise OnnxExportError(
                f"Batch size {batch_size} produced invalid output shape {output.shape}."
            )
        finite = bool(np.isfinite(output).all())
        if not finite:
            raise OnnxExportError(f"Batch size {batch_size} produced non-finite logits.")
        results.append(
            OnnxBatchValidation(
                batch_size=batch_size,
                output_shape=tuple(int(dimension) for dimension in output.shape),
                all_finite=finite,
            )
        )

    return OnnxValidationReport(
        onnx_path=str(path),
        input_name=inputs[0].name,
        output_name=outputs[0].name,
        dynamic_input_batch=dynamic_input,
        dynamic_output_batch=dynamic_output,
        providers=tuple(session.get_providers()),
        batches=tuple(results),
    )


def load_resnet18_model(
    weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
) -> nn.Module:
    """Load the production ResNet18 on demand and place it in evaluation mode."""

    model = cast(nn.Module, resnet18(weights=weights))
    model.eval()
    return model


def export_resnet18_to_onnx(
    output_path: str | Path,
    *,
    weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
    batch_sizes: Sequence[int] = (1, 4, 16),
    opset_version: int = 18,
) -> OnnxValidationReport:
    """Load, export, check, and execute the production ResNet18 artifact."""

    model = load_resnet18_model(weights)
    path = export_model_to_onnx(model, output_path, opset_version=opset_version)
    return validate_onnx_model(path, batch_sizes=batch_sizes, input_shape=(3, 224, 224))


# Compact aliases for CLI and test callers.
export_to_onnx = export_model_to_onnx
export_resnet18 = export_resnet18_to_onnx
validate_onnx_export = validate_onnx_model


__all__ = [
    "OnnxBatchValidation",
    "OnnxExportError",
    "OnnxValidationReport",
    "export_model_to_onnx",
    "export_resnet18",
    "export_resnet18_to_onnx",
    "export_to_onnx",
    "load_resnet18_model",
    "validate_onnx_export",
    "validate_onnx_model",
]

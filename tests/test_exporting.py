from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

import inference_gateway.exporting as exporting
from inference_gateway.exporting import (
    OnnxBatchValidation,
    OnnxExportError,
    OnnxValidationReport,
    export_model_to_onnx,
    validate_onnx_model,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 8 * 8, 7)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))


@pytest.fixture(scope="module")
def exported_tiny_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TinyClassifier, Path]:
    torch.manual_seed(17)
    model = TinyClassifier().eval()
    path = tmp_path_factory.mktemp("export") / "tiny.onnx"
    exported = export_model_to_onnx(model, path, torch.zeros(1, 3, 8, 8))
    return model, exported


def test_dynamo_export_has_symbolic_input_and_output_batch_axes(
    exported_tiny_model: tuple[TinyClassifier, Path],
) -> None:
    _, path = exported_tiny_model
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    input_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    output_dimension = model.graph.output[0].type.tensor_type.shape.dim[0]

    assert model.graph.input[0].name == "images"
    assert model.graph.output[0].name == "logits"
    assert input_dimension.WhichOneof("value") == "dim_param"
    assert output_dimension.WhichOneof("value") == "dim_param"
    assert input_dimension.dim_param
    assert output_dimension.dim_param


def test_export_loads_in_ort_and_executes_required_batch_sizes(
    exported_tiny_model: tuple[TinyClassifier, Path],
) -> None:
    _, path = exported_tiny_model
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(23)

    for batch_size in (1, 4, 16):
        values = rng.standard_normal((batch_size, 3, 8, 8), dtype=np.float32)
        output = session.run(["logits"], {"images": values})[0]
        assert output.shape == (batch_size, 7)


def test_validation_runs_checker_and_real_inference(
    exported_tiny_model: tuple[TinyClassifier, Path],
) -> None:
    _, path = exported_tiny_model

    report = validate_onnx_model(path, batch_sizes=(1, 4, 16))

    assert report.passed
    assert report.dynamic_input_batch
    assert report.dynamic_output_batch
    assert report.batch_sizes == (1, 4, 16)
    assert [batch.output_shape for batch in report.batches] == [
        (1, 7),
        (4, 7),
        (16, 7),
    ]
    assert report.to_dict()["passed"] is True
    assert report.batches[0].to_dict()["output_shape"] == [1, 7]


def test_validation_rejects_a_static_batch_axis(
    exported_tiny_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    _, dynamic_path = exported_tiny_model
    model = onnx.load(str(dynamic_path))
    for value_info in (model.graph.input[0], model.graph.output[0]):
        dimension = value_info.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_param")
        dimension.dim_value = 1
    static_path = tmp_path / "static.onnx"
    onnx.save(model, str(static_path))

    with pytest.raises(OnnxExportError, match="dynamic input and output batch axes"):
        validate_onnx_model(static_path)


def test_validation_rejects_missing_corrupt_and_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_onnx_model(tmp_path / "missing.onnx")

    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an ONNX model")
    with pytest.raises(OnnxExportError, match="checker rejected"):
        validate_onnx_model(corrupt)

    with pytest.raises(ValueError, match="positive integers"):
        validate_onnx_model(corrupt, batch_sizes=())


def test_export_validates_its_example_and_wraps_exporter_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyClassifier().eval()
    with pytest.raises(ValueError, match="sample_input"):
        export_model_to_onnx(model, tmp_path / "bad-rank.onnx", torch.zeros(1))
    with pytest.raises(ValueError, match="input_shape"):
        export_model_to_onnx(model, tmp_path / "bad-shape.onnx", input_shape=(3, -1, 8))

    def fail_export(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic exporter failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    with pytest.raises(OnnxExportError, match="Failed to export") as error:
        export_model_to_onnx(model, tmp_path / "failure.onnx", torch.zeros(1, 3, 8, 8))
    assert isinstance(error.value.__cause__, RuntimeError)


def test_export_rejects_a_noop_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)

    with pytest.raises(OnnxExportError, match="did not create"):
        export_model_to_onnx(
            TinyClassifier().eval(),
            tmp_path / "absent.onnx",
            torch.zeros(1, 3, 8, 8),
        )


def test_resnet_helpers_can_be_exercised_without_real_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny = TinyClassifier().eval()
    monkeypatch.setattr(exporting, "resnet18", lambda *, weights: tiny)

    assert exporting.load_resnet18_model(weights=None) is tiny

    sentinel = OnnxValidationReport(
        onnx_path=str(tmp_path / "tiny.onnx"),
        input_name="images",
        output_name="logits",
        dynamic_input_batch=True,
        dynamic_output_batch=True,
        providers=("CPUExecutionProvider",),
        batches=(OnnxBatchValidation(1, (1, 7), True),),
    )

    def fake_export(
        _model: nn.Module,
        output_path: str | Path,
        **_kwargs: object,
    ) -> Path:
        return Path(output_path)

    monkeypatch.setattr(exporting, "load_resnet18_model", lambda _weights: tiny)
    monkeypatch.setattr(exporting, "export_model_to_onnx", fake_export)
    monkeypatch.setattr(exporting, "validate_onnx_model", lambda *_args, **_kwargs: sentinel)

    assert exporting.export_resnet18_to_onnx(tmp_path / "tiny.onnx", weights=None) is sentinel

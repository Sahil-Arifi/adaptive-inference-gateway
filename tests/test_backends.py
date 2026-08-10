from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from inference_gateway.backends import FakeBackend, InferenceBackend, OnnxBackend, TorchBackend
from inference_gateway.backends.base import (
    BackendClosedError,
    BackendConfigurationError,
    BackendExecutionError,
    validate_batch,
)
from inference_gateway.backends.onnx_backend import resolve_onnx_providers
from inference_gateway.backends.torch_backend import resolve_torch_device
from inference_gateway.exporting import export_model_to_onnx


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 5 * 5, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))


@pytest.fixture(scope="module")
def local_model_and_onnx(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TinyClassifier, Path]:
    torch.manual_seed(31)
    model = TinyClassifier().eval()
    path = tmp_path_factory.mktemp("backend") / "tiny.onnx"
    export_model_to_onnx(model, path, torch.zeros(1, 3, 5, 5))
    return model, path


def test_fake_backend_satisfies_protocol_and_is_deterministic() -> None:
    backend = FakeBackend(num_classes=6)
    batch = np.stack(
        [
            np.zeros((3, 4, 4), dtype=np.float32),
            np.ones((3, 4, 4), dtype=np.float32),
        ]
    )

    first = backend.predict_logits(batch)
    second = backend.predict_logits(batch.copy())

    assert isinstance(backend, InferenceBackend)
    assert backend.name == "fake"
    assert backend.device == "cpu"
    assert first.shape == (2, 6)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert backend.call_count == 2
    assert backend.inference_calls == 2
    assert backend.batch_sizes == (2, 2)


def test_fake_backend_warmup_failure_and_close_paths() -> None:
    backend = FakeBackend(num_classes=2)
    assert not backend.warmed_up
    backend.warmup()
    assert backend.warmed_up
    backend.close()
    backend.close()

    with pytest.raises(BackendClosedError, match="closed"):
        backend.warmup()
    with pytest.raises(BackendClosedError, match="closed"):
        backend.predict_logits(np.zeros((1, 3), dtype=np.float32))

    failing = FakeBackend(num_classes=2, fail=True)
    with pytest.raises(RuntimeError, match="Configured fake backend failure"):
        failing.predict_logits(np.zeros((1, 3), dtype=np.float32))
    assert failing.call_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"num_classes": 0}, "positive"), ({"delay_seconds": -0.1}, "negative")],
)
def test_fake_backend_validates_configuration(kwargs: dict[str, int | float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FakeBackend(**kwargs)  # type: ignore[arg-type]


def test_batch_validation_normalizes_numeric_arrays_and_rejects_invalid_inputs() -> None:
    normalized = validate_batch(np.ones((2, 3), dtype=np.float64)[:, ::-1])
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous

    with pytest.raises(TypeError, match="NumPy"):
        validate_batch([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch and feature"):
        validate_batch(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        validate_batch(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="numeric"):
        validate_batch(np.array([["image"]], dtype=object))


def test_torch_backend_runs_local_model_and_matches_direct_pytorch(
    local_model_and_onnx: tuple[TinyClassifier, Path],
) -> None:
    model, _ = local_model_and_onnx
    backend = TorchBackend(model=model, device="cpu", input_shape=(3, 5, 5))
    rng = np.random.default_rng(37)
    batch = rng.standard_normal((4, 3, 5, 5), dtype=np.float32)

    actual = backend.predict_logits(batch)
    with torch.inference_mode():
        expected = model(torch.from_numpy(batch)).numpy()

    assert isinstance(backend, InferenceBackend)
    assert backend.name == "torch"
    assert backend.device == "cpu"
    assert backend.model is model
    assert not model.training
    assert actual.shape == (4, 6)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    backend.warmup()
    backend.close()
    backend.close()
    with pytest.raises(BackendClosedError, match="closed"):
        backend.predict_logits(batch)


def test_torch_backend_rejects_invalid_configuration_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto").type == "cpu"
    assert resolve_torch_device(" CPU ").type == "cpu"
    with pytest.raises(BackendConfigurationError, match="CUDA was requested"):
        resolve_torch_device("cuda")
    with pytest.raises(BackendConfigurationError, match="must be"):
        resolve_torch_device("metal")
    with pytest.raises(ValueError, match="input_shape"):
        TorchBackend(model=TinyClassifier(), input_shape=())
    with pytest.raises(ValueError, match="warmup_batch_size"):
        TorchBackend(model=TinyClassifier(), warmup_batch_size=0)

    class TupleModel(nn.Module):
        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
            return (images,)

    tuple_backend = TorchBackend(model=TupleModel(), device="cpu", input_shape=(3, 5, 5))
    with pytest.raises(BackendExecutionError, match="did not return a tensor"):
        tuple_backend.predict_logits(np.zeros((1, 3, 5, 5), dtype=np.float32))

    class WrongBatchModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros((images.shape[0] + 1, 2), device=images.device)

    wrong_backend = TorchBackend(model=WrongBatchModel(), device="cpu", input_shape=(3, 5, 5))
    with pytest.raises(BackendExecutionError, match="invalid batch shape"):
        wrong_backend.predict_logits(np.zeros((1, 3, 5, 5), dtype=np.float32))


def test_onnx_backend_matches_torch_and_closes_cleanly(
    local_model_and_onnx: tuple[TinyClassifier, Path],
) -> None:
    model, path = local_model_and_onnx
    backend = OnnxBackend(path, device="cpu")
    rng = np.random.default_rng(41)
    batch = rng.standard_normal((4, 3, 5, 5), dtype=np.float32)

    actual = backend.predict_logits(batch)
    with torch.inference_mode():
        expected = model(torch.from_numpy(batch)).numpy()

    assert isinstance(backend, InferenceBackend)
    assert backend.name == "onnx"
    assert backend.device == "cpu"
    assert backend.model_path == path
    assert backend.input_name == "images"
    assert backend.output_name == "logits"
    assert "CPUExecutionProvider" in backend.session.get_providers()
    assert actual.shape == (4, 6)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    backend.warmup()
    backend.close()
    backend.close()
    with pytest.raises(BackendClosedError, match="closed"):
        _ = backend.session
    with pytest.raises(BackendClosedError, match="closed"):
        backend.warmup()


def test_onnx_backend_validates_paths_shapes_and_providers(
    local_model_and_onnx: tuple[TinyClassifier, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, path = local_model_and_onnx
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OnnxBackend(tmp_path / "missing.onnx")
    with pytest.raises(ValueError, match="warmup_batch_size"):
        OnnxBackend(path, device="cpu", warmup_batch_size=0)
    with pytest.raises(ValueError, match="input_shape"):
        OnnxBackend(path, device="cpu", input_shape=(3, -1, 5))

    monkeypatch.setattr(
        "inference_gateway.backends.onnx_backend.ort.get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    assert resolve_onnx_providers("auto") == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda",
    )
    assert resolve_onnx_providers("cuda")[1] == "cuda"

    monkeypatch.setattr(
        "inference_gateway.backends.onnx_backend.ort.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    assert resolve_onnx_providers("auto") == (["CPUExecutionProvider"], "cpu")
    with pytest.raises(BackendConfigurationError, match="CUDA was requested"):
        resolve_onnx_providers("cuda")

    monkeypatch.setattr(
        "inference_gateway.backends.onnx_backend.ort.get_available_providers",
        lambda: [],
    )
    with pytest.raises(BackendConfigurationError, match="no supported"):
        resolve_onnx_providers("auto")
    with pytest.raises(BackendConfigurationError, match="not installed"):
        resolve_onnx_providers("cpu")
    with pytest.raises(BackendConfigurationError, match="must be"):
        resolve_onnx_providers("tpu")


def test_onnx_backend_rejects_invalid_runtime_output(
    local_model_and_onnx: tuple[TinyClassifier, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, path = local_model_and_onnx
    backend = OnnxBackend(path, device="cpu")
    batch = np.zeros((2, 3, 5, 5), dtype=np.float32)

    def wrong_output(
        _output_names: list[str],
        _input_feed: dict[str, npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        return [np.zeros((1, 6), dtype=np.float32)]

    monkeypatch.setattr(backend.session, "run", wrong_output)
    with pytest.raises(BackendExecutionError, match="invalid batch shape"):
        backend.predict_logits(batch)

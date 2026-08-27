from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import onnxruntime as ort
import pytest
import torch
from torch import nn

from inference_gateway.exporting import export_model_to_onnx
from inference_gateway.parity import (
    PARITY_SCHEMA_VERSION,
    ParityReport,
    ParityVerificationError,
    load_parity_report,
    require_passing_parity,
    sha256_file,
    validate_production_parity_policy,
    verify_and_write_parity,
    verify_parity,
    write_parity_report,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 6 * 6, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))


class LightweightProductionContract(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(self.pool(images)))


@pytest.fixture(scope="module")
def parity_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[TinyClassifier, Path]:
    torch.manual_seed(101)
    model = TinyClassifier().eval()
    path = tmp_path_factory.mktemp("parity") / "tiny.onnx"
    export_model_to_onnx(model, path, torch.zeros(1, 3, 6, 6))
    return model, path


@pytest.fixture
def passing_report(parity_model: tuple[TinyClassifier, Path]) -> ParityReport:
    model, path = parity_model
    return verify_parity(model, path, batch_sizes=(1, 4, 16), seed=404)


@pytest.fixture(scope="module")
def production_policy_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ParityReport, Path]:
    torch.manual_seed(2028)
    model = LightweightProductionContract().eval()
    path = tmp_path_factory.mktemp("production-policy") / "resnet18-contract.onnx"
    export_model_to_onnx(model, path, torch.zeros(1, 3, 224, 224))
    report = verify_parity(
        model,
        path,
        batch_sizes=(1, 4, 16),
        input_shape=(3, 224, 224),
        rtol=1e-4,
        atol=1e-5,
        seed=2029,
        device="cpu",
        model_name="resnet18",
    )
    return report, path


def test_exact_input_parity_passes_with_per_batch_metrics(
    passing_report: ParityReport,
) -> None:
    report = passing_report

    assert report.schema_version == PARITY_SCHEMA_VERSION
    assert report.batch_sizes == (1, 4, 16)
    assert report.passed
    assert report.max_abs_difference < 1e-4
    assert report.mean_abs_difference < report.max_abs_difference
    assert report.top1_agreement == 1.0
    assert all(batch.allclose and batch.passed for batch in report.per_batch)
    assert all(batch.top1_agreement == 1.0 for batch in report.per_batch)


def test_parity_is_deterministic_for_a_fixed_seed(
    parity_model: tuple[TinyClassifier, Path],
) -> None:
    model, path = parity_model

    first = verify_parity(model, path, batch_sizes=(1, 4), seed=99)
    second = verify_parity(model, path, batch_sizes=(1, 4), seed=99)

    assert first.per_batch == second.per_batch
    assert first.max_abs_difference == second.max_abs_difference
    assert first.mean_abs_difference == second.mean_abs_difference


def test_parity_rejects_silent_cuda_provider_fallback_before_model_execution(
    parity_model: tuple[TinyClassifier, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, path = parity_model
    cpu_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(ort, "InferenceSession", lambda *_args, **_kwargs: cpu_session)

    with pytest.raises(ParityVerificationError, match="did not activate"):
        verify_parity(model, path, batch_sizes=(1,), device="cuda")

    assert next(model.parameters()).device.type == "cpu"


def test_report_round_trip_and_matching_sha_gate(
    passing_report: ParityReport,
    tmp_path: Path,
) -> None:
    report_path = write_parity_report(passing_report, tmp_path / "nested" / "parity.json")

    loaded = load_parity_report(report_path)

    assert loaded == passing_report
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_production_cpu_policy_and_matching_sha_gate(
    production_policy_report: tuple[ParityReport, Path],
    tmp_path: Path,
) -> None:
    report, model_path = production_policy_report
    report_path = write_parity_report(report, tmp_path / "production-parity.json")

    trusted = require_passing_parity(report_path, model_path)

    assert trusted == report
    assert trusted.onnx_sha256 == sha256_file(model_path)


def test_production_policy_device_is_selected_by_the_caller(
    production_policy_report: tuple[ParityReport, Path],
) -> None:
    report, _ = production_policy_report
    cuda_report = replace(report, device="cuda")

    validate_production_parity_policy(cuda_report, expected_device="cuda")


def test_stale_hash_is_rejected(
    production_policy_report: tuple[ParityReport, Path],
    tmp_path: Path,
) -> None:
    report, model_path = production_policy_report
    stale = replace(report, onnx_sha256="0" * 64)
    report_path = write_parity_report(stale, tmp_path / "stale.json")

    with pytest.raises(ParityVerificationError, match="does not match"):
        require_passing_parity(report_path, model_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_name": "ResNet"}, "model identity"),
        ({"input_shape": (3, 112, 112)}, "input shape"),
        ({"rtol": 1e-3}, "rtol"),
        ({"atol": 1e-4}, "atol"),
        ({"device": "cuda"}, "device"),
    ],
)
def test_production_policy_rejects_tampered_identity_shape_tolerance_and_device(
    production_policy_report: tuple[ParityReport, Path],
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    report, model_path = production_policy_report
    tampered = replace(report, **changes)
    report_path = write_parity_report(tampered, tmp_path / f"tampered-{message}.json")

    with pytest.raises(ParityVerificationError, match=message):
        require_passing_parity(report_path, model_path)


def test_production_policy_requires_exact_batch_order(
    production_policy_report: tuple[ParityReport, Path],
    tmp_path: Path,
) -> None:
    report, model_path = production_policy_report
    reordered = replace(
        report,
        per_batch=(report.per_batch[1], report.per_batch[0], report.per_batch[2]),
    )
    report_path = write_parity_report(reordered, tmp_path / "reordered-batches.json")

    with pytest.raises(ParityVerificationError, match="batch sizes must be exactly"):
        require_passing_parity(report_path, model_path)


def test_production_policy_requires_one_thousand_logits_per_image(
    production_policy_report: tuple[ParityReport, Path],
    tmp_path: Path,
) -> None:
    report, model_path = production_policy_report
    batches = tuple(
        replace(batch, logit_count=batch.batch_size * 999) for batch in report.per_batch
    )
    total_logits = sum(batch.logit_count for batch in batches)
    aggregate_mean = (
        sum(batch.mean_abs_difference * batch.logit_count for batch in batches)
        / total_logits
    )
    tampered = replace(report, per_batch=batches, mean_abs_difference=aggregate_mean)
    report_path = write_parity_report(tampered, tmp_path / "wrong-output-size.json")

    with pytest.raises(ParityVerificationError, match="1000 logits per image"):
        require_passing_parity(report_path, model_path)


def test_failed_parity_artifact_is_rejected(
    passing_report: ParityReport,
    parity_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    _, model_path = parity_model
    first = replace(passing_report.per_batch[0], allclose=False, passed=False)
    failed = replace(
        passing_report,
        per_batch=(first, *passing_report.per_batch[1:]),
        passed=False,
    )
    report_path = write_parity_report(failed, tmp_path / "failed.json")

    with pytest.raises(ParityVerificationError, match="records a failed"):
        require_passing_parity(report_path, model_path)


@pytest.mark.parametrize("contents", ["not json", "[]", "{}"])
def test_malformed_parity_artifacts_fail_closed(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ParityVerificationError, match="Invalid parity report"):
        load_parity_report(path)


def test_internally_inconsistent_parity_artifact_fails_closed(
    passing_report: ParityReport,
    tmp_path: Path,
) -> None:
    data = passing_report.to_dict()
    data["passed"] = False
    path = tmp_path / "inconsistent.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ParityVerificationError, match="Invalid parity report"):
        load_parity_report(path)


def test_missing_evidence_or_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ParityVerificationError, match="report does not exist"):
        load_parity_report(tmp_path / "missing.json")

    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ParityVerificationError, match="ONNX model does not exist"):
        require_passing_parity(report_path, tmp_path / "missing.onnx")


def test_verify_and_write_records_a_failed_comparison_before_raising(
    parity_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    _, path = parity_model
    torch.manual_seed(999)
    different_model = TinyClassifier().eval()
    destination = tmp_path / "failed-parity.json"

    with pytest.raises(ParityVerificationError, match="verification failed"):
        verify_and_write_parity(
            different_model,
            path,
            destination,
            batch_sizes=(1, 4),
            seed=707,
        )

    assert destination.is_file()
    assert load_parity_report(destination).passed is False


def test_verify_parity_rejects_invalid_arguments_and_outputs(
    parity_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    model, path = parity_model
    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_parity(model, tmp_path / "missing.onnx")
    with pytest.raises(ValueError, match="positive integers"):
        verify_parity(model, path, batch_sizes=(0,))
    with pytest.raises(ValueError, match="cannot be negative"):
        verify_parity(model, path, rtol=-1)
    with pytest.raises(ValueError, match="input_shape"):
        verify_parity(model, path, input_shape=(3, -1, 6))

    class TupleModel(nn.Module):
        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
            return (images,)

    with pytest.raises(ParityVerificationError, match="did not return a tensor"):
        verify_parity(TupleModel(), path, batch_sizes=(1,))

    class WrongShapeModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros((images.shape[0], 3), device=images.device)

    with pytest.raises(ParityVerificationError, match="shape mismatch"):
        verify_parity(WrongShapeModel(), path, batch_sizes=(1,))

    class NonFiniteModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.full((images.shape[0], 5), float("inf"), device=images.device)

    with pytest.raises(ParityVerificationError, match="non-finite"):
        verify_parity(NonFiniteModel(), path, batch_sizes=(1,))

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

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


def test_report_round_trip_and_matching_sha_gate(
    passing_report: ParityReport,
    parity_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    _, model_path = parity_model
    report_path = write_parity_report(passing_report, tmp_path / "nested" / "parity.json")

    loaded = load_parity_report(report_path)
    trusted = require_passing_parity(report_path, model_path)

    assert loaded == passing_report
    assert trusted == passing_report
    assert trusted.onnx_sha256 == sha256_file(model_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_stale_hash_is_rejected(
    passing_report: ParityReport,
    parity_model: tuple[TinyClassifier, Path],
    tmp_path: Path,
) -> None:
    _, model_path = parity_model
    stale = replace(passing_report, onnx_sha256="0" * 64)
    report_path = write_parity_report(stale, tmp_path / "stale.json")

    with pytest.raises(ParityVerificationError, match="does not match"):
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

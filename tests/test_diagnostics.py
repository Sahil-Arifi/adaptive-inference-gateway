from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from inference_gateway import diagnostics
from inference_gateway.cli import app
from inference_gateway.diagnostics import (
    DiagnosticPlan,
    run_diagnostics,
    stage_summary,
    summarize_trials,
)
from inference_gateway.loadgen import LoadTestResult, RequestSample, _optional_float


def result() -> LoadTestResult:
    sample = RequestSample(
        0,
        0,
        0.020,
        200,
        True,
        False,
        None,
        "a",
        2.0,
        4.0,
        1,
        server_processing_ms=15.0,
        upload_and_parse_ms=3.0,
        preprocessing_ms=5.0,
    )
    return LoadTestResult(1, 1, 0, 0, 0, 0.020, 50.0, 20.0, 20.0, 20.0, 20.0, (sample,))


def test_stage_residuals_and_missing_or_failed_telemetry() -> None:
    stats = stage_summary(result())
    assert stats["client_minus_server_ms"] == 5
    assert stats["preprocessing_ms"] == 5
    missing = replace(result(), samples=(replace(result().samples[0], server_processing_ms=None),))
    assert stage_summary(missing)["client_minus_server_ms"] is None
    failed = replace(result(), samples=(replace(result().samples[0], success=False),))
    assert all(value is None for value in stage_summary(failed).values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency": ()},
        {"concurrency": (0,)},
        {"concurrency": (1, 1)},
        {"repetitions": 1},
        {"requests": 1},
        {"warmup_requests": -1},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_plans_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DiagnosticPlan(**kwargs)


def test_trial_order_and_aggregation_are_repeatable() -> None:
    plan = DiagnosticPlan()
    assert plan.order() == DiagnosticPlan().order()
    assert len(set(plan.order())) == 20
    trials = [
        {"concurrency": 8, "result": result().to_dict()},
        {
            "concurrency": 8,
            "result": replace(result(), throughput_requests_per_second=70).to_dict(),
        },
    ]
    aggregate = summarize_trials(trials)[0]["metrics"]["throughput_requests_per_second"]
    assert aggregate["mean"] == 60
    assert aggregate["stdev"] == pytest.approx(14.1421356237)
    assert (aggregate["min"], aggregate["max"]) == (50, 70)
    trials[0]["result"]["p95_latency_ms"] = None
    trials[1]["result"]["p95_latency_ms"] = None
    assert summarize_trials(trials)[0]["metrics"]["p95_latency_ms"]["mean"] is None


class FakeGenerator:
    changed = False
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> FakeGenerator:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def warmup(self, *args: Any) -> None:
        pass

    async def get_json(self, *args: Any) -> dict[str, Any]:
        type(self).calls += 1
        return {
            "backend": "fake" if not self.changed or self.calls <= 2 else "changed",
            "device": "cpu",
            "scheduler_mode": "direct",
        }

    async def run(self, *args: Any) -> LoadTestResult:
        return result()


@pytest.mark.asyncio
async def test_raw_trials_saved_without_replacing_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "AsyncLoadGenerator", FakeGenerator)
    plan = DiagnosticPlan(concurrency=(1, 2), repetitions=2, requests=2, warmup_requests=0)
    output = tmp_path / "diagnostic"
    summary = await run_diagnostics("http://fake", output, plan)
    assert summary["completed_trials"] == 4
    assert len(list(output.glob("trial-*.json"))) == 4
    raw = json.loads((output / "trial-001.json").read_text())
    assert raw["stage_means"]["client_minus_server_ms"] == 5
    with pytest.raises(FileExistsError):
        await run_diagnostics("http://fake", output, plan)


@pytest.mark.asyncio
async def test_changed_server_is_rejected_and_completed_trial_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "AsyncLoadGenerator", FakeGenerator)
    monkeypatch.setattr(FakeGenerator, "changed", True)
    monkeypatch.setattr(FakeGenerator, "calls", 0)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="changed between trials"):
        await run_diagnostics("http://fake", output, DiagnosticPlan(concurrency=(1,), requests=1))
    assert (output / "trial-001.json").exists()
    assert not (output / "summary.json").exists()


def test_cli_validation_and_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    args = [
        "diagnose",
        "--output",
        str(tmp_path / "cli"),
        "--concurrency",
        "1",
        "--requests",
        "1",
        "--repetitions",
        "2",
    ]
    monkeypatch.setattr(diagnostics, "AsyncLoadGenerator", FakeGenerator)
    response = runner.invoke(app, args)
    assert response.exit_code == 0, response.output
    assert "Completed 2 trials" in response.output
    assert runner.invoke(app, args).exit_code != 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "bad"])
def test_invalid_timing_telemetry_is_not_reported(value: Any) -> None:
    assert _optional_float({"time": value}, "time") is None


@pytest.mark.asyncio
async def test_identity_change_in_last_trial_and_incomplete_stats_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LastSnapshotChanges(FakeGenerator):
        async def get_json(self, *args: Any) -> dict[str, Any]:
            type(self).calls += 1
            return {
                "backend": "changed" if self.calls == 4 else "fake",
                "device": "cpu",
                "scheduler_mode": "direct",
            }

    monkeypatch.setattr(LastSnapshotChanges, "calls", 0)
    monkeypatch.setattr(diagnostics, "AsyncLoadGenerator", LastSnapshotChanges)
    with pytest.raises(ValueError, match="during a trial"):
        await run_diagnostics(
            "http://fake",
            tmp_path / "last",
            DiagnosticPlan(concurrency=(1,), requests=1, repetitions=2),
        )
    assert (tmp_path / "last" / "trial-001.json").exists()
    assert not (tmp_path / "last" / "summary.json").exists()
    with pytest.raises(ValueError, match="must identify"):
        diagnostics._server_identity({"backend": "fake"})

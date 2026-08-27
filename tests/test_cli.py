from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from inference_gateway import cli
from inference_gateway.loadgen import LoadTestResult
from inference_gateway.reporting import ArtifactPaths

runner = CliRunner()


@dataclass
class SerializableReport:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def test_cli_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    commands = ("export", "parity", "serve", "make-demo-image", "loadtest", "benchmark", "report")
    for command in commands:
        assert command in result.stdout


def test_make_demo_image_command(tmp_path: Path) -> None:
    output = tmp_path / "demo.png"

    result = runner.invoke(cli.app, ["make-demo-image", str(output)])

    assert result.exit_code == 0
    assert output.read_bytes().startswith(b"\x89PNG")


def test_export_command_renders_validation_report(monkeypatch: Any) -> None:
    report = SerializableReport({"passed": True, "batch_sizes": [1, 4, 16]})
    monkeypatch.setattr(cli, "export_resnet18_to_onnx", lambda path: report)

    result = runner.invoke(cli.app, ["export", "--config", "configs/default.yaml"])

    assert result.exit_code == 0
    assert '"passed": true' in result.stdout
    assert '"batch_sizes"' in result.stdout


def test_parity_command_writes_configured_artifact(monkeypatch: Any) -> None:
    report = SerializableReport({"passed": True, "top1_agreement": 1.0})
    observed: dict[str, Any] = {}
    model = object()
    monkeypatch.setattr(cli, "load_resnet18_model", lambda: model)

    def fake_verify(*args: Any, **kwargs: Any) -> SerializableReport:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli, "verify_and_write_parity", fake_verify)

    result = runner.invoke(cli.app, ["parity", "--config", "configs/default.yaml"])

    assert result.exit_code == 0
    assert observed["args"][0] is model
    assert observed["args"][2] == Path("artifacts/parity.json")
    assert observed["kwargs"]["device"] == "auto"
    assert observed["kwargs"]["model_name"] == "resnet18"
    assert '"top1_agreement": 1.0' in result.stdout


def test_serve_command_uses_validated_host_and_port(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}
    fake_app = object()
    monkeypatch.setattr(cli, "create_app", lambda settings: fake_app)

    def fake_run(app: object, **kwargs: Any) -> None:
        observed["app"] = app
        observed.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--config", "configs/default.yaml"])

    assert result.exit_code == 0
    assert observed == {
        "app": fake_app,
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
    }


def test_loadtest_command_renders_measured_values(monkeypatch: Any) -> None:
    async def fake_loadtest(*args: Any, **kwargs: Any) -> LoadTestResult:
        assert args == ("http://example.test",)
        assert kwargs["requests"] == 12
        assert kwargs["concurrency"] == 3
        return LoadTestResult(
            requested=12,
            successful_requests=10,
            failed_requests=2,
            http_429_responses=1,
            timed_out_requests=1,
            duration_seconds=1.0,
            throughput_requests_per_second=10.0,
            mean_latency_ms=4.0,
            p50_latency_ms=3.0,
            p95_latency_ms=7.0,
            p99_latency_ms=8.0,
            samples=(),
        )

    monkeypatch.setattr(cli, "run_load_test", fake_loadtest)

    result = runner.invoke(
        cli.app,
        [
            "loadtest",
            "--url",
            "http://example.test",
            "--requests",
            "12",
            "--concurrency",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Requests/s" in result.stdout
    assert "10.0" in result.stdout


def test_benchmark_command_reports_case_count(monkeypatch: Any) -> None:
    suite = SimpleNamespace(cases=tuple(range(32)))
    observed: dict[str, Any] = {}

    def fake_benchmark(settings: Any, *, update_readme: bool) -> Any:
        observed["count"] = settings.benchmark.primary_case_count
        observed["update"] = update_readme
        return suite

    monkeypatch.setattr(cli, "run_benchmark_sync", fake_benchmark)

    result = runner.invoke(
        cli.app,
        ["benchmark", "--config", "configs/benchmark.yaml", "--no-update-readme"],
    )

    assert result.exit_code == 0
    assert observed == {"count": 32, "update": False}
    assert "Completed 32 primary cases" in result.stdout


def test_report_command_renders_written_paths(monkeypatch: Any, tmp_path: Path) -> None:
    artifacts = ArtifactPaths(
        results_json=tmp_path / "results.json",
        results_csv=tmp_path / "results.csv",
        report_markdown=tmp_path / "report.md",
        throughput_vs_p95_chart=tmp_path / "throughput.png",
        batch_efficiency_chart=tmp_path / "batch.png",
    )
    observed: dict[str, Any] = {}

    def fake_report(path: Path, *, update_readme: bool) -> ArtifactPaths:
        observed["path"] = path
        observed["update"] = update_readme
        return artifacts

    monkeypatch.setattr(cli, "generate_report", fake_report)

    result = runner.invoke(
        cli.app,
        ["report", "--results", "custom.json", "--no-update-readme"],
    )

    assert result.exit_code == 0
    assert observed == {"path": Path("custom.json"), "update": False}
    assert "report.md" in result.stdout

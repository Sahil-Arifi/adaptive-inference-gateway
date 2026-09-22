"""Command-line interface for setup, serving, load generation, and benchmarks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from inference_gateway.benchmark import run_benchmark_sync
from inference_gateway.config import load_config
from inference_gateway.exporting import export_resnet18_to_onnx, load_resnet18_model
from inference_gateway.loadgen import run_load_test, write_deterministic_image
from inference_gateway.parity import verify_and_write_parity
from inference_gateway.reporting import generate_report
from inference_gateway.service import create_app

app = typer.Typer(
    name="inference-gateway",
    help="Export, verify, serve, and benchmark the adaptive image inference gateway.",
    no_args_is_help=True,
)
console = Console()


@app.command("export")
def export_command(
    config: Annotated[
        Path,
        typer.Option(
        "--config",
        help="Validated gateway YAML configuration.",
        ),
    ] = Path("configs/default.yaml"),
) -> None:
    """Download ResNet18 weights, export dynamic ONNX, and validate batches 1/4/16."""

    settings = load_config(config)
    report = export_resnet18_to_onnx(settings.model.onnx_path)
    console.print_json(data=report.to_dict())


@app.command("parity")
def parity_command(
    config: Annotated[
        Path,
        typer.Option(
        "--config",
        help="Validated gateway YAML configuration.",
        ),
    ] = Path("configs/default.yaml"),
) -> None:
    """Compare deterministic PyTorch/ONNX logits and write artifacts/parity.json."""

    settings = load_config(config)
    model = load_resnet18_model()
    destination = settings.output.directory / "parity.json"
    report = verify_and_write_parity(
        model,
        settings.model.onnx_path,
        destination,
        device=settings.model.device.value,
        model_name=settings.model.architecture,
    )
    console.print_json(data=report.to_dict())


@app.command("serve")
def serve_command(
    config: Annotated[
        Path,
        typer.Option(
        "--config",
        help="Validated gateway YAML configuration.",
        ),
    ] = Path("configs/default.yaml"),
) -> None:
    """Start the FastAPI gateway after backend warmup and scheduler startup."""

    settings = load_config(config)
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
    )


@app.command("make-demo-image")
def make_demo_image_command(
    output: Annotated[
        Path,
        typer.Argument(help="Output path for the deterministic PNG image."),
    ] = Path("artifacts/demo.png"),
) -> None:
    """Generate one deterministic RGB image; no external dataset is required."""

    path = write_deterministic_image(output)
    console.print(f"Wrote deterministic demo image to [bold]{path}[/bold]")


@app.command("loadtest")
def loadtest_command(
    url: Annotated[
        str,
        typer.Option("--url", help="Base URL of a running gateway."),
    ] = "http://127.0.0.1:8000",
    requests: Annotated[
        int,
        typer.Option("--requests", min=1, help="Measured request count."),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", min=1, help="Concurrent workers."),
    ] = 32,
    warmup_requests: Annotated[
        int,
        typer.Option(
            "--warmup-requests",
            min=0,
            help="Unmeasured requests sent before measurement.",
        ),
    ] = 0,
    timeout_seconds: Annotated[
        float,
        typer.Option(
            "--timeout-seconds",
            min=0.001,
            help="Per-request client timeout.",
        ),
    ] = 30.0,
) -> None:
    """Run a real concurrent HTTP load test with individual latency samples."""

    result = asyncio.run(
        run_load_test(
            url,
            requests=requests,
            concurrency=concurrency,
            warmup_requests=warmup_requests,
            timeout_seconds=timeout_seconds,
        )
    )
    table = Table(title="Load-test result")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = (
        ("Requests", result.requested),
        ("Success", result.successful_requests),
        ("Failures", result.failed_requests),
        ("HTTP 429", result.http_429_responses),
        ("Timeouts", result.timed_out_requests),
        ("Requests/s", result.throughput_requests_per_second),
        ("Mean latency ms", result.mean_latency_ms),
        ("p50 latency ms", result.p50_latency_ms),
        ("p95 latency ms", result.p95_latency_ms),
        ("p99 latency ms", result.p99_latency_ms),
    )
    for label, value in rows:
        display = "n/a" if value is None else str(value)
        table.add_row(label, display)
    console.print(table)


@app.command("benchmark")
def benchmark_command(
    config: Annotated[
        Path,
        typer.Option(
        "--config",
        help="Benchmark YAML defining request counts and the 32-case matrix.",
        ),
    ] = Path("configs/benchmark.yaml"),
    update_readme: Annotated[
        bool,
        typer.Option(
            "--update-readme/--no-update-readme",
            help="Replace the marked README benchmark section from measured results.",
        ),
    ] = True,
) -> None:
    """Run all 32 CPU cases over loopback HTTP and generate benchmark artifacts."""

    settings = load_config(config)
    suite = run_benchmark_sync(settings, update_readme=update_readme)
    console.print(
        f"Completed [bold]{len(suite.cases)}[/bold] primary cases; "
        f"artifacts are in [bold]{settings.output.directory}[/bold]."
    )


@app.command("report")
def report_command(
    results: Annotated[
        Path,
        typer.Option(
        "--results",
        help="Canonical full-precision benchmark results JSON.",
        ),
    ] = Path("artifacts/results.json"),
    update_readme: Annotated[
        bool,
        typer.Option(
            "--update-readme/--no-update-readme",
            help="Replace the marked README benchmark section from these results.",
        ),
    ] = True,
) -> None:
    """Regenerate CSV, Markdown, and charts exclusively from results.json."""

    artifacts = generate_report(results, update_readme=update_readme)
    console.print_json(
        data={
            "results_json": str(artifacts.results_json),
            "results_csv": str(artifacts.results_csv),
            "report": str(artifacts.report_markdown),
            "throughput_chart": str(artifacts.throughput_vs_p95_chart),
            "batch_chart": str(artifacts.batch_efficiency_chart),
        }
    )


@app.command("diagnose")
def diagnose_command(
    output: Annotated[Path, typer.Option("--output", help="New diagnostics directory.")],
    url: Annotated[str, typer.Option("--url")] = "http://127.0.0.1:8000",
    concurrency: Annotated[str, typer.Option("--concurrency")] = "1,8,32,64",
    repetitions: Annotated[int, typer.Option("--repetitions", min=2)] = 5,
    requests: Annotated[int, typer.Option("--requests", min=1)] = 2000,
    warmup_requests: Annotated[int, typer.Option("--warmup-requests", min=0)] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 2027,
) -> None:
    """Repeat a randomized concurrency sweep against a running server; save raw trials."""
    from inference_gateway.diagnostics import DiagnosticPlan, run_diagnostics

    try:
        plan = DiagnosticPlan(
            concurrency=tuple(int(value.strip()) for value in concurrency.split(",")),
            repetitions=repetitions, requests=requests, warmup_requests=warmup_requests, seed=seed,
        )
        result = asyncio.run(run_diagnostics(url, output, plan))
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Completed {result['completed_trials']} trials; raw evidence: {output}")


def main() -> None:
    """Console-script adapter kept separate for direct module execution."""

    app()


if __name__ == "__main__":
    main()

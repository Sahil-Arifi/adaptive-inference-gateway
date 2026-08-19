"""Data-derived benchmark serialization, Markdown reporting, and charts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)


README_RESULTS_START = "<!-- BENCHMARK_RESULTS_START -->"
README_RESULTS_END = "<!-- BENCHMARK_RESULTS_END -->"

_CSV_FIELDS = (
    "case_id",
    "backend",
    "device",
    "scheduler_mode",
    "concurrency",
    "max_batch_size",
    "max_wait_ms",
    "requests",
    "successful_requests",
    "failures",
    "rejections",
    "timeouts",
    "throughput_requests_per_second",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_queue_wait_ms",
    "p95_queue_wait_ms",
    "mean_backend_inference_ms",
    "mean_realized_batch_size",
    "maximum_realized_batch_size",
    "backend_inference_call_count",
    "batches_executed",
    "duration_seconds",
)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Paths written from one canonical results payload."""

    results_json: Path
    results_csv: Path
    report_markdown: Path
    throughput_vs_p95_chart: Path
    batch_efficiency_chart: Path


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a benchmark JSON object for report regeneration."""

    results_path = Path(path)
    with results_path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"expected a JSON object in {results_path}")
    _cases(payload)
    return payload


def _cases(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = results.get("cases")
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("benchmark results must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(candidate):
        if not isinstance(case, dict) or not all(isinstance(key, str) for key in case):
            raise ValueError(f"benchmark case {index} is not a JSON object")
        cases.append(case)
    return cases


def _number(case: Mapping[str, Any], key: str) -> float | None:
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(case: Mapping[str, Any], key: str) -> int:
    value = case.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _text(case: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = case.get(key)
    return value if isinstance(value, str) else default


def _format_number(value: Any, *, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _best_case(
    cases: Sequence[Mapping[str, Any]],
    key: str,
    *,
    maximize: bool,
) -> Mapping[str, Any] | None:
    candidates = [
        case
        for case in cases
        if _integer(case, "successful_requests") > 0 and _number(case, key) is not None
    ]
    if not candidates:
        return None
    return (max if maximize else min)(
        candidates,
        key=lambda case: float(_number(case, key) or 0.0),
    )


def _case_label(case: Mapping[str, Any]) -> str:
    return _text(case, "case_id", "unnamed case")


def _json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _csv_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = {field: case.get(field) for field in _CSV_FIELDS}
        delta = case.get("server_stats_delta")
        if isinstance(delta, dict):
            for key, value in delta.items():
                if isinstance(key, str) and not isinstance(value, (dict, list)):
                    row[f"stats_delta_{key}"] = value
        rows.append(row)
    return rows


def _write_csv(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(_csv_rows(cases))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _group_plot_points(
    cases: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> dict[str, list[tuple[float, float, str]]]:
    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for case in cases:
        x_value = _number(case, x_key)
        y_value = _number(case, y_key)
        if x_value is None or y_value is None:
            continue
        label = f"{_text(case, 'backend')} / {_text(case, 'scheduler_mode')}"
        groups[label].append((x_value, y_value, _case_label(case)))
    return groups


def _save_scatter(
    groups: Mapping[str, Sequence[tuple[float, float, str]]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    if groups:
        for index, (label, points) in enumerate(sorted(groups.items())):
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                label=label,
                marker=markers[index % len(markers)],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.6,
                s=58,
            )
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No successful benchmark measurements",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": "adaptive-inference-gateway"})
    plt.close(fig)


def _write_charts(cases: Sequence[Mapping[str, Any]], output_directory: Path) -> tuple[Path, Path]:
    throughput_path = output_directory / "throughput_vs_p95.png"
    batch_path = output_directory / "batch_efficiency.png"
    _save_scatter(
        _group_plot_points(
            cases,
            x_key="p95_latency_ms",
            y_key="throughput_requests_per_second",
        ),
        x_label="p95 latency (ms)",
        y_label="Throughput (successful requests/s)",
        title="Throughput versus tail latency",
        path=throughput_path,
    )
    _save_scatter(
        _group_plot_points(
            cases,
            x_key="mean_realized_batch_size",
            y_key="throughput_requests_per_second",
        ),
        x_label="Mean realized batch size",
        y_label="Throughput (successful requests/s)",
        title="Realized batching efficiency",
        path=batch_path,
    )
    return throughput_path, batch_path


def _environment_lines(environment: Mapping[str, Any]) -> list[str]:
    fields = (
        ("Operating system", "operating_system"),
        ("Machine", "machine"),
        ("Processor", "processor"),
        ("Logical CPUs", "logical_cpu_count"),
        ("Python", "python_version"),
        ("PyTorch", "torch_version"),
        ("torchvision", "torchvision_version"),
        ("ONNX", "onnx_version"),
        ("ONNX Runtime", "onnxruntime_version"),
        ("CUDA available", "cuda_available"),
        ("CUDA version", "cuda_version"),
        ("CUDA device", "cuda_device_name"),
    )
    return [f"- {label}: `{environment.get(key, 'unknown')}`" for label, key in fields]


def _results_table(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Case | Device | C | Batch | Wait ms | Success | Reject | Timeout | req/s | "
        "Mean ms | p50 ms | p95 ms | p99 ms | Mean queue ms | Mean backend ms | "
        "Mean realized batch | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                (
                    _case_label(case),
                    _text(case, "device"),
                    str(_integer(case, "concurrency")),
                    str(_integer(case, "max_batch_size")),
                    _format_number(case.get("max_wait_ms"), digits=1),
                    str(_integer(case, "successful_requests")),
                    str(_integer(case, "rejections")),
                    str(_integer(case, "timeouts")),
                    _format_number(case.get("throughput_requests_per_second")),
                    _format_number(case.get("mean_latency_ms")),
                    _format_number(case.get("p50_latency_ms")),
                    _format_number(case.get("p95_latency_ms")),
                    _format_number(case.get("p99_latency_ms")),
                    _format_number(case.get("mean_queue_wait_ms")),
                    _format_number(case.get("mean_backend_inference_ms")),
                    _format_number(case.get("mean_realized_batch_size")),
                    str(_integer(case, "backend_inference_call_count")),
                )
            )
            + " |"
        )
    return lines


def _batching_tradeoff_lines(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for backend in ("torch", "onnx"):
        for concurrency in sorted({_integer(case, "concurrency") for case in cases}):
            direct = [
                case
                for case in cases
                if _text(case, "backend") == backend
                and _text(case, "scheduler_mode") == "direct"
                and _integer(case, "concurrency") == concurrency
            ]
            dynamic = [
                case
                for case in cases
                if _text(case, "backend") == backend
                and _text(case, "scheduler_mode") == "dynamic"
                and _integer(case, "concurrency") == concurrency
            ]
            if not direct or not dynamic:
                continue
            best_dynamic = _best_case(
                dynamic,
                "throughput_requests_per_second",
                maximize=True,
            )
            direct_case = direct[0]
            if best_dynamic is None:
                continue
            direct_throughput = _number(direct_case, "throughput_requests_per_second")
            dynamic_throughput = _number(best_dynamic, "throughput_requests_per_second")
            direct_p95 = _number(direct_case, "p95_latency_ms")
            dynamic_p95 = _number(best_dynamic, "p95_latency_ms")
            if (
                direct_throughput is None
                or dynamic_throughput is None
                or direct_p95 is None
                or dynamic_p95 is None
            ):
                continue
            throughput_change = (
                100.0 * (dynamic_throughput - direct_throughput) / direct_throughput
                if direct_throughput != 0.0
                else 0.0
            )
            p95_change = (
                100.0 * (dynamic_p95 - direct_p95) / direct_p95
                if direct_p95 != 0.0
                else 0.0
            )
            lines.append(
                f"- {backend} at concurrency {concurrency}: best-throughput dynamic case "
                f"`{_case_label(best_dynamic)}` changed throughput by "
                f"{throughput_change:+.2f}% and p95 latency by {p95_change:+.2f}% "
                "relative to direct."
            )
    if not lines:
        lines.append("- No matched successful direct/dynamic measurements were available.")
    return lines


def _backend_comparison_lines(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    direct_cases = [case for case in cases if _text(case, "scheduler_mode") == "direct"]
    for concurrency in sorted({_integer(case, "concurrency") for case in direct_cases}):
        torch_cases = [
            case
            for case in direct_cases
            if _text(case, "backend") == "torch"
            and _integer(case, "concurrency") == concurrency
        ]
        onnx_cases = [
            case
            for case in direct_cases
            if _text(case, "backend") == "onnx"
            and _integer(case, "concurrency") == concurrency
        ]
        if not torch_cases or not onnx_cases:
            continue
        torch_throughput = _number(torch_cases[0], "throughput_requests_per_second")
        onnx_throughput = _number(onnx_cases[0], "throughput_requests_per_second")
        torch_p95 = _number(torch_cases[0], "p95_latency_ms")
        onnx_p95 = _number(onnx_cases[0], "p95_latency_ms")
        if (
            torch_throughput is None
            or onnx_throughput is None
            or torch_p95 is None
            or onnx_p95 is None
        ):
            continue
        lines.append(
            f"- Direct concurrency {concurrency}: PyTorch "
            f"{torch_throughput:.3f} req/s at {torch_p95:.3f} ms p95; "
            f"ONNX {onnx_throughput:.3f} req/s at {onnx_p95:.3f} ms p95."
        )
    if not lines:
        lines.append("- No matched successful PyTorch/ONNX direct measurements were available.")
    return lines


def render_report(results: Mapping[str, Any]) -> str:
    """Render a Markdown report solely from measured result data."""

    cases = _cases(results)
    environment_candidate = results.get("environment", {})
    environment = (
        environment_candidate if isinstance(environment_candidate, dict) else {}
    )
    config_candidate = results.get("benchmark_config", {})
    config = config_candidate if isinstance(config_candidate, dict) else {}
    parity_candidate = results.get("parity_artifact", {})
    parity = parity_candidate if isinstance(parity_candidate, dict) else {}

    best_throughput = _best_case(
        cases,
        "throughput_requests_per_second",
        maximize=True,
    )
    best_p95 = _best_case(cases, "p95_latency_ms", maximize=False)
    best_lines = []
    if best_throughput is not None:
        best_lines.append(
            f"- Best measured throughput: `{_case_label(best_throughput)}` at "
            f"{_format_number(best_throughput.get('throughput_requests_per_second'))} "
            "successful requests/s."
        )
    if best_p95 is not None:
        best_lines.append(
            f"- Best measured p95 latency: `{_case_label(best_p95)}` at "
            f"{_format_number(best_p95.get('p95_latency_ms'))} ms."
        )
    if not best_lines:
        best_lines.append("- No successful measurements were available.")

    benchmark_config = config.get("benchmark", {})
    scheduler_config = config.get("scheduler", {})
    requests_per_case = (
        benchmark_config.get("requests_per_case", "unknown")
        if isinstance(benchmark_config, dict)
        else "unknown"
    )
    warmup_requests = (
        benchmark_config.get("warmup_requests", "unknown")
        if isinstance(benchmark_config, dict)
        else "unknown"
    )
    request_timeout_ms = (
        scheduler_config.get("request_timeout_ms", "unknown")
        if isinstance(scheduler_config, dict)
        else "unknown"
    )
    lines = [
        "# Adaptive Inference Gateway Benchmark Report",
        "",
        f"Generated: `{results.get('generated_at_utc', 'unknown')}`",
        "",
        "## Environment",
        "",
        *_environment_lines(environment),
        "",
        "## Experiment configuration",
        "",
        f"- Completed primary cases: `{len(cases)}`",
        f"- Requests per case: `{requests_per_case}`",
        f"- Warmup requests per case: `{warmup_requests}`",
        f"- Request timeout ms: `{request_timeout_ms}`",
        "- Inputs: the fixed synthetic image sequence recorded by SHA-256 in `results.json`.",
        "- Latency percentiles: calculated from individual successful HTTP request "
        "samples; warmups are excluded.",
        "- Throughput: successful measured responses divided by measured wall-clock duration.",
        "",
        "## ONNX parity gate",
        "",
        f"- Passed: `{parity.get('passed', 'unknown')}`",
        f"- ONNX SHA-256: `{parity.get('onnx_sha256', 'unknown')}`",
        f"- Maximum absolute logit difference: `{parity.get('max_abs_difference', 'unknown')}`",
        f"- Mean absolute logit difference: `{parity.get('mean_abs_difference', 'unknown')}`",
        f"- Top-1 agreement: `{parity.get('top1_agreement', 'unknown')}`",
        "",
        "## Highlights",
        "",
        *best_lines,
        "",
        "## Full primary results",
        "",
        *_results_table(cases),
        "",
        "## Observed batching trade-offs",
        "",
        *_batching_tradeoff_lines(cases),
        "",
        "Positive changes mean an increase; a throughput increase and a p95 increase "
        "therefore describe a throughput/latency trade-off, not an unqualified improvement.",
        "",
        "## PyTorch versus ONNX",
        "",
        *_backend_comparison_lines(cases),
        "",
        "No backend is assumed faster; the statements above are generated from this run.",
        "",
        "## Charts",
        "",
        "![Throughput versus p95 latency](throughput_vs_p95.png)",
        "",
        "![Realized batch size versus throughput](batch_efficiency.png)",
        "",
        "## Limitations",
        "",
        "- The synthetic inputs make serving runs reproducible but do not measure "
        "ImageNet accuracy.",
        "- These results describe only the recorded hardware, software versions, "
        "and configuration.",
        "- Client and server share one host, so contention and loopback transport "
        "affect measurements.",
        "- This educational scheduler demonstrates production concepts; it is not "
        "a replacement for NVIDIA Triton.",
        "- Results from differently labeled CPU and CUDA runs must not be combined "
        "as if they were one environment.",
        "",
    ]
    return "\n".join(lines)


def _readme_excerpt(results: Mapping[str, Any], output_directory: Path, readme_path: Path) -> str:
    cases = _cases(results)
    best_throughput = _best_case(
        cases,
        "throughput_requests_per_second",
        maximize=True,
    )
    best_p95 = _best_case(cases, "p95_latency_ms", maximize=False)
    throughput_chart = os.path.relpath(
        output_directory / "throughput_vs_p95.png",
        readme_path.parent,
    ).replace("\\", "/")
    batch_chart = os.path.relpath(
        output_directory / "batch_efficiency.png",
        readme_path.parent,
    ).replace("\\", "/")
    report_path = os.path.relpath(
        output_directory / "report.md",
        readme_path.parent,
    ).replace("\\", "/")
    lines = [
        README_RESULTS_START,
        "## Measured benchmark results",
        "",
        f"This benchmark completed {len(cases)} primary cases on the environment "
        "recorded in `artifacts/results.json`.",
    ]
    if best_throughput is not None:
        lines.extend(
            (
                "",
                f"- Best throughput: `{_case_label(best_throughput)}` — "
                f"{_format_number(best_throughput.get('throughput_requests_per_second'))} req/s",
            )
        )
    if best_p95 is not None:
        lines.extend(
            (
                f"- Best p95 latency: `{_case_label(best_p95)}` — "
                f"{_format_number(best_p95.get('p95_latency_ms'))} ms",
                "",
            )
        )
    lines.extend(
        (
            f"![Throughput versus p95 latency]({throughput_chart})",
            "",
            f"![Realized batch size versus throughput]({batch_chart})",
            "",
            f"See [`{report_path}`]({report_path}) for the full generated table and "
            "methodology.",
            README_RESULTS_END,
        )
    )
    return "\n".join(lines)


def update_marked_readme(
    readme_path: str | Path,
    results: Mapping[str, Any],
    output_directory: str | Path,
) -> Path:
    """Replace an explicitly marked README section with measured results."""

    path = Path(readme_path)
    source = path.read_text(encoding="utf-8")
    start = source.find(README_RESULTS_START)
    end = source.find(README_RESULTS_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            f"{path} must contain ordered {README_RESULTS_START} and {README_RESULTS_END} markers"
        )
    end += len(README_RESULTS_END)
    replacement = _readme_excerpt(results, Path(output_directory), path)
    updated = source[:start] + replacement + source[end:]
    path.write_text(updated, encoding="utf-8", newline="\n")
    return path


def write_benchmark_artifacts(
    results: Mapping[str, Any],
    output_directory: str | Path,
    *,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> ArtifactPaths:
    """Write JSON, CSV, Markdown, and both charts from one results object."""

    cases = _cases(results)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    results_json = output_path / "results.json"
    results_csv = output_path / "results.csv"
    report_path = output_path / "report.md"

    _json_dump(results, results_json)
    _write_csv(cases, results_csv)
    throughput_chart, batch_chart = _write_charts(cases, output_path)
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    if update_readme:
        update_marked_readme(readme_path, results, output_path)
    return ArtifactPaths(
        results_json=results_json,
        results_csv=results_csv,
        report_markdown=report_path,
        throughput_vs_p95_chart=throughput_chart,
        batch_efficiency_chart=batch_chart,
    )


def generate_report(
    results_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    update_readme: bool = False,
    readme_path: str | Path = "README.md",
) -> ArtifactPaths:
    """Regenerate all derived artifacts from an existing results.json."""

    source_path = Path(results_path)
    results = load_results(source_path)
    destination = Path(output_directory) if output_directory is not None else source_path.parent
    return write_benchmark_artifacts(
        results,
        destination,
        update_readme=update_readme,
        readme_path=readme_path,
    )


__all__ = [
    "README_RESULTS_END",
    "README_RESULTS_START",
    "ArtifactPaths",
    "generate_report",
    "load_results",
    "render_report",
    "update_marked_readme",
    "write_benchmark_artifacts",
]

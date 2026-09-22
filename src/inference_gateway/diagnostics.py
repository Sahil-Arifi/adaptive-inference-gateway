"""Repeated, randomized concurrency experiments against an explicitly running server."""

from __future__ import annotations

import asyncio
import json
import math
import platform
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inference_gateway.loadgen import AsyncLoadGenerator, LoadTestResult, generate_synthetic_images


@dataclass(frozen=True)
class DiagnosticPlan:
    concurrency: tuple[int, ...] = (1, 8, 32, 64)
    repetitions: int = 5
    requests: int = 2000
    warmup_requests: int = 100
    seed: int = 2027
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.concurrency or any(c < 1 for c in self.concurrency):
            raise ValueError("concurrency values must be positive")
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("concurrency values must be unique")
        if self.repetitions < 2 or self.requests < max(self.concurrency):
            raise ValueError("use at least two repetitions and requests >= maximum concurrency")
        if (
            self.warmup_requests < 0
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("invalid warmup count or timeout")

    def order(self) -> list[tuple[int, int]]:
        rng = random.Random(self.seed)
        ordered: list[tuple[int, int]] = []
        for repetition in range(self.repetitions):
            levels = list(self.concurrency)
            rng.shuffle(levels)
            ordered.extend((repetition + 1, concurrency) for concurrency in levels)
        return ordered


def stage_summary(result: LoadTestResult) -> dict[str, float | None]:
    """Means from successful samples only. Residuals are not isolated network latency."""
    successful = [sample for sample in result.samples if sample.success]
    fields = (
        "server_processing_ms",
        "upload_and_parse_ms",
        "preprocessing_ms",
        "queue_wait_ms",
        "backend_inference_ms",
        "realized_batch_size",
    )
    summary: dict[str, float | None] = {}
    for field in fields:
        values = [getattr(sample, field) for sample in successful]
        summary[field] = (
            statistics.fmean(value for value in values if value is not None)
            if values and all(value is not None for value in values)
            else None
        )
    residuals = [
        sample.elapsed_seconds * 1000 - sample.server_processing_ms
        for sample in successful
        if sample.server_processing_ms is not None
    ]
    summary["client_minus_server_ms"] = (
        statistics.fmean(residuals) if successful and len(residuals) == len(successful) else None
    )
    return summary


def summarize_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report variability across whole trials, never pretend requests are independent runs."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(trial["concurrency"], []).append(trial)
    summary = []
    for concurrency, group in sorted(grouped.items()):
        metrics = {}
        for metric in ("throughput_requests_per_second", "p95_latency_ms", "p99_latency_ms"):
            values = [
                trial["result"][metric] for trial in group if trial["result"][metric] is not None
            ]
            metrics[metric] = {
                "trials_with_value": len(values),
                "mean": statistics.fmean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        summary.append({"concurrency": concurrency, "repetitions": len(group), "metrics": metrics})
    return summary


def _server_identity(snapshot: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(snapshot.get(key) for key in ("backend", "device", "scheduler_mode"))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("server stats must identify backend, device, and scheduler_mode")
    return tuple(str(value) for value in values)


async def run_diagnostics(base_url: str, destination: Path, plan: DiagnosticPlan) -> dict[str, Any]:
    """Persist every completed trial immediately; never update canonical benchmark results."""
    await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=False)
    images = generate_synthetic_images()
    trials: list[dict[str, Any]] = []
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "client_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "plan": asdict(plan),
        "trial_order": plan.order(),
        "limitations": [
            "Closed-loop workers bound concurrency; this is not a fixed arrival-rate workload.",
            "Tail latencies describe successful requests; failures remain in each raw trial.",
            "Client minus server time includes transport, response handling, "
            "and timer boundary differences.",
            "No confidence interval or causal bottleneck claim is inferred from these diagnostics.",
        ],
    }
    await asyncio.to_thread(
        (destination / "plan.json").write_text,
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    identity = None
    for index, (repetition, concurrency) in enumerate(plan.order(), 1):
        async with AsyncLoadGenerator(
            base_url, concurrency=concurrency, timeout_seconds=plan.timeout_seconds
        ) as generator:
            await generator.warmup(images, plan.warmup_requests)
            before = await generator.get_json("/stats")
            current = _server_identity(before)
            if identity is not None and current != identity:
                raise ValueError("server backend/device/scheduler changed between trials")
            identity = current
            result = await generator.run(images, plan.requests)
            after = await generator.get_json("/stats")
            if _server_identity(after) != current:
                raise ValueError("server backend/device/scheduler changed during a trial")
        trial = {
            "repetition": repetition,
            "concurrency": concurrency,
            "server_stats_before": before,
            "server_stats_after": after,
            "stage_means": stage_summary(result),
            "result": result.to_dict(),
        }
        await asyncio.to_thread(
            (destination / f"trial-{index:03d}.json").write_text,
            json.dumps(trial, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        trials.append(trial)
    output = {**manifest, "completed_trials": len(trials), "summary": summarize_trials(trials)}
    await asyncio.to_thread(
        (destination / "summary.json").write_text,
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output

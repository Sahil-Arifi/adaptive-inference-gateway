"""Send one image to a running adaptive inference gateway."""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="JPEG or PNG image to classify")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="gateway base URL (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def call_gateway(image_path: Path, url: str, timeout: float) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as image_file:
        response = httpx.post(
            f"{url.rstrip('/')}/v1/predict",
            files={"file": (image_path.name, image_file, media_type)},
            timeout=timeout,
        )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


def render(payload: dict[str, Any]) -> None:
    console = Console()
    table = Table(title="ImageNet predictions")
    table.add_column("Rank", justify="right")
    table.add_column("Class")
    table.add_column("Index", justify="right")
    table.add_column("Confidence", justify="right")
    for rank, prediction in enumerate(payload["predictions"], start=1):
        table.add_row(
            str(rank),
            str(prediction["label"]),
            str(prediction["index"]),
            f"{float(prediction['confidence']):.4%}",
        )
    console.print(table)
    console.print(
        f"request={payload['request_id']} backend={payload['backend']} "
        f"device={payload['device']} scheduler={payload['scheduler_mode']} "
        f"batch={payload['realized_batch_size']} "
        f"server={float(payload['server_processing_ms']):.2f} ms"
    )


def main() -> None:
    args = parse_args()
    try:
        render(call_gateway(args.image, args.url, args.timeout))
    except (FileNotFoundError, httpx.HTTPError, KeyError, ValueError) as exc:
        raise SystemExit(f"demo failed: {exc}") from exc


if __name__ == "__main__":
    main()

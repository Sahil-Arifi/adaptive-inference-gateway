"""Deterministic, dependency-light backend used by unit tests."""

from __future__ import annotations

import threading
import time

import numpy as np

from inference_gateway.backends.base import BackendClosedError, FloatBatch, validate_batch


class FakeBackend:
    """Generate deterministic logits while recording physical backend calls."""

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        delay_seconds: float = 0.0,
        fail: bool = False,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative.")
        self.num_classes = num_classes
        self.delay_seconds = delay_seconds
        self.fail = fail
        self._closed = False
        self._warmed_up = False
        self._call_count = 0
        self._batch_sizes: list[int] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def inference_calls(self) -> int:
        """Alias that reads naturally in scheduler statistics and tests."""

        return self.call_count

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._batch_sizes)

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    def predict_logits(self, batch: np.ndarray) -> FloatBatch:
        values = validate_batch(batch)
        with self._lock:
            if self._closed:
                raise BackendClosedError("The fake backend is closed.")
            self._call_count += 1
            self._batch_sizes.append(int(values.shape[0]))

        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("Configured fake backend failure.")

        flattened = values.reshape(values.shape[0], -1)
        indices = np.linspace(
            0,
            flattened.shape[1] - 1,
            num=self.num_classes,
            dtype=np.int64,
        )
        sampled = flattened[:, indices]
        sample_means = flattened.mean(axis=1, dtype=np.float64).astype(np.float32)[:, None]
        class_offsets = np.linspace(-0.5, 0.5, self.num_classes, dtype=np.float32)[None, :]
        logits = sampled * np.float32(0.5) + sample_means + class_offsets
        return np.ascontiguousarray(logits, dtype=np.float32)

    def warmup(self) -> None:
        with self._lock:
            if self._closed:
                raise BackendClosedError("The fake backend is closed.")
            self._warmed_up = True

    def close(self) -> None:
        with self._lock:
            self._closed = True


__all__ = ["FakeBackend"]

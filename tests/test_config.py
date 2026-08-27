from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_gateway.config import (
    BackendName,
    GatewaySettings,
    SchedulerMode,
    dump_config,
    load_config,
)


@pytest.mark.parametrize(
    "config_path",
    ["configs/default.yaml", "configs/benchmark.yaml", "configs/docker-smoke.yaml"],
)
def test_repository_configs_load_required_benchmark_matrix(config_path: str) -> None:
    settings = load_config(config_path)

    assert settings.benchmark.primary_case_count == 32
    assert settings.benchmark.dynamic_concurrency == [8, 32, 64]


def test_default_config_uses_onnx_dynamic_scheduler() -> None:
    settings = load_config("configs/default.yaml")

    assert settings.model.backend is BackendName.ONNX
    assert settings.scheduler.mode is SchedulerMode.DYNAMIC


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file does not exist"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "[]\n"])
def test_config_root_must_be_mapping(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)


def test_unknown_configuration_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GatewaySettings.model_validate({"server": {"prt": 9000}})


def test_dynamic_collection_window_must_precede_timeout() -> None:
    with pytest.raises(ValidationError, match="max_wait_ms must be shorter"):
        GatewaySettings.model_validate(
            {
                "scheduler": {
                    "mode": "dynamic",
                    "max_wait_ms": 10,
                    "request_timeout_ms": 10,
                }
            }
        )


def test_invalid_benchmark_matrix_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly 32 cases"):
        GatewaySettings.model_validate({"benchmark": {"concurrency": [1, 8]}})


def test_dumped_config_round_trips(tmp_path: Path) -> None:
    expected = load_config("configs/default.yaml")
    output = dump_config(expected, tmp_path / "nested" / "config.yaml")

    actual = load_config(output)

    assert actual == expected


def test_empty_yaml_uses_valid_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    settings = load_config(path)

    assert settings.model.architecture == "resnet18"
    assert settings.benchmark.primary_case_count == 32

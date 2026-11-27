# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup partitions must reach the runtime, or fail before dispatch."""

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.timing.config import TimingConfig

_CLI_PARITY_YAML = """schemaVersion: '2.0'
benchmark:
  model: test
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    entries: 12
  warmup:
    type: concurrency
    concurrency: 1
    requests: 4
  profiling:
    type: concurrency
    concurrency: 1
    requests: 8
"""


_PARTITIONED_YAML = """schemaVersion: '2.0'
benchmark:
  model: test
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    entries: 12
    warmup_entries: 4
  profiling:
    type: concurrency
    concurrency: 1
    requests: 8
"""


def test_cli_and_yaml_partition_resolve_to_same_runtime(tmp_path: Path) -> None:
    """Keep CLI, YAML overrides, and the runtime reservation in agreement."""
    cli = CLIConfig(
        model_names=["test"],
        request_count=8,
        warmup_request_count=4,
        conversation_num_dataset_entries=12,
        warmup_dataset_entries=4,
    )
    config = resolve_config(cli)
    run = BenchmarkRun(
        benchmark_id="test",
        cfg=config.benchmark,
        artifact_dir=tmp_path,
    )
    assert TimingConfig.from_run(run).warmup_dataset_entries == 4
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        _CLI_PARITY_YAML,
        encoding="utf-8",
    )
    from_yaml = resolve_config(CLIConfig(warmup_dataset_entries=4), config_file)
    assert from_yaml.benchmark.get_default_dataset().warmup_entries == 4
    assert from_yaml.benchmark.get_profiling_phases()[0].requests == 8
    config_file.write_text(
        _CLI_PARITY_YAML.replace(
            "    entries: 12", "    entries: 12\n    warmup_entries: 4"
        ),
        encoding="utf-8",
    )
    disabled = resolve_config(CLIConfig(warmup_dataset_entries=0), config_file)
    assert disabled.benchmark.get_default_dataset().warmup_entries == 0


@pytest.mark.parametrize(
    "overrides,match",
    [
        param({"warmup_dataset_entries": -1}, "greater than or equal", id="negative"),
        param({"warmup_dataset_entries": 12}, "smaller than", id="empty_profile"),
    ],
)  # fmt: skip
def test_invalid_partition_rejected_before_dispatch(
    overrides: dict, match: str
) -> None:
    """Reject invalid reservation sizes before scheduling work."""
    args = dict(
        model_names=["test"],
        request_count=8,
        warmup_request_count=4,
        conversation_num_dataset_entries=12,
        warmup_dataset_entries=4,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match=match):
        resolve_config(CLIConfig(**args))


def test_default_pool_remains_shared() -> None:
    """Preserve shared sampling when no reservation is requested."""
    config = resolve_config(CLIConfig(model_names=["test"], request_count=8))
    assert config.benchmark.get_default_dataset().warmup_entries == 0


def test_partition_requires_warmup() -> None:
    """Reject authored partitions without a warmup phase."""
    with pytest.raises(ValueError, match="requires a warmup"):
        resolve_config(
            CLIConfig(model_names=["test"], request_count=8, warmup_dataset_entries=2)
        )


def test_file_dataset_flag_is_rejected_by_name(tmp_path: Path) -> None:
    """Name the unsupported CLI flag on a file-backed dataset."""
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "hello"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="--warmup-dataset-entries"):
        resolve_config(
            CLIConfig(
                model_names=["test"],
                input_file=str(dataset),
                warmup_dataset_entries=2,
                warmup_request_count=2,
                request_count=8,
            )
        )


@pytest.fixture
def partitioned_yaml(tmp_path: Path) -> Path:
    """Provide an authored reservation whose warmup can come from CLI."""
    path = tmp_path / "partitioned.yaml"
    path.write_text(
        _PARTITIONED_YAML,
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "overrides,reserved,warmup_requests",
    [
        param({"warmup_request_count": 4}, 4, 4, id="add_warmup"),
        param({"warmup_dataset_entries": 0}, 0, None, id="disable_partition"),
    ],
)  # fmt: skip
def test_yaml_partition_eligibility_uses_final_cli_config(
    partitioned_yaml: Path,
    overrides: dict,
    reserved: int,
    warmup_requests: int | None,
) -> None:
    """Apply added warmup or explicit zero before final eligibility checks."""
    config = resolve_config(CLIConfig(**overrides), partitioned_yaml)
    benchmark = config.benchmark
    run = BenchmarkRun(
        benchmark_id="test", cfg=benchmark, artifact_dir=partitioned_yaml.parent
    )
    assert TimingConfig.from_run(run).warmup_dataset_entries == reserved
    assert [phase.requests for phase in benchmark.get_warmup_phases()] == (
        [warmup_requests] if warmup_requests is not None else []
    )
    assert benchmark.get_profiling_phases()[0].requests == 8


def test_cli_zero_can_disable_yaml_partition_at_pool_boundary(
    partitioned_yaml: Path,
) -> None:
    """Allow CLI zero to remove a reservation that exhausts the YAML pool."""
    partitioned_yaml.write_text(
        _PARTITIONED_YAML.replace("entries: 12", "entries: 4"),
        encoding="utf-8",
    )
    config = resolve_config(CLIConfig(warmup_dataset_entries=0), partitioned_yaml)
    assert config.benchmark.get_default_dataset().warmup_entries == 0
    assert config.benchmark.get_default_dataset().entries == 4


def test_yaml_partition_still_requires_warmup_after_merge(
    partitioned_yaml: Path,
) -> None:
    """Keep the authored missing-warmup guard after preliminary validation."""
    with pytest.raises(ValueError, match="requires a warmup phase"):
        resolve_config(CLIConfig(), partitioned_yaml)


@pytest.mark.parametrize(
    "old,new,match",
    [
        param("entries: 12", "entries: 4", "must be smaller", id="pool_boundary"),
        param(
            "requests: 8", "requests: 8\n    timing_mode: agentic_replay",
            "not supported with", id="replay_continuity",
        ),
    ],
)  # fmt: skip
def test_final_partition_constraints_are_not_skipped(
    partitioned_yaml: Path,
    old: str,
    new: str,
    match: str,
) -> None:
    """Enforce pool size and replay restrictions after all CLI overrides."""
    partitioned_yaml.write_text(
        _PARTITIONED_YAML.replace(old, new),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        resolve_config(CLIConfig(warmup_request_count=4), partitioned_yaml)


@pytest.mark.parametrize(
    "source",
    ["cli", "yaml_partition_cli_accuracy", "yaml_accuracy_cli_warmup"],
)
def test_accuracy_partition_rejected_before_dataset_loading(
    partitioned_yaml: Path,
    source: str,
) -> None:
    """Reject benchmark-owned accuracy data instead of partitioning its answers."""
    args = {"warmup_request_count": 4}
    if source == "cli":
        args.update(
            model_names=["test"],
            request_count=8,
            conversation_num_dataset_entries=12,
            warmup_dataset_entries=4,
            accuracy_benchmark="mmlu",
        )
        path = None
    else:
        path = partitioned_yaml
        if source == "yaml_partition_cli_accuracy":
            args["accuracy_benchmark"] = "mmlu"
        else:
            path.write_text(
                _PARTITIONED_YAML + "  accuracy:\n    benchmark: mmlu\n",
                encoding="utf-8",
            )
    with pytest.raises(
        ValueError, match="--warmup-dataset-entries.*--accuracy-benchmark"
    ):
        resolve_config(CLIConfig(**args), path)


@pytest.mark.parametrize(
    "benchmark,reserved",
    [
        param(None, 4, id="accuracy_disabled"),
        param("mmlu", 0, id="explicit_zero_with_accuracy"),
    ],
)  # fmt: skip
def test_accuracy_partition_guard_preserves_supported_configs(
    partitioned_yaml: Path,
    benchmark: str | None,
    reserved: int,
) -> None:
    """Keep disabled accuracy and explicit-zero configurations valid."""
    partitioned_yaml.write_text(
        _PARTITIONED_YAML
        + f"  accuracy:\n    benchmark: {benchmark if benchmark else 'null'}\n",
        encoding="utf-8",
    )
    args = {"warmup_dataset_entries": reserved}
    if reserved:
        args["warmup_request_count"] = 4
    config = resolve_config(CLIConfig(**args), partitioned_yaml)
    assert config.benchmark.get_default_dataset().warmup_entries == reserved
    assert config.benchmark.accuracy.enabled is (benchmark is not None)

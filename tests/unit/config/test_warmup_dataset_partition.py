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


def test_cli_and_yaml_partition_resolve_to_same_runtime(tmp_path: Path) -> None:
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
        """schemaVersion: '2.0'
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
""",
        encoding="utf-8",
    )
    from_yaml = resolve_config(CLIConfig(warmup_dataset_entries=4), config_file)
    assert from_yaml.benchmark.get_default_dataset().warmup_entries == 4
    assert from_yaml.benchmark.get_profiling_phases()[0].requests == 8
    config_file.write_text(
        config_file.read_text().replace(
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
    config = resolve_config(CLIConfig(model_names=["test"], request_count=8))
    assert config.benchmark.get_default_dataset().warmup_entries == 0


def test_partition_requires_warmup() -> None:
    with pytest.raises(ValueError, match="requires a warmup"):
        resolve_config(
            CLIConfig(model_names=["test"], request_count=8, warmup_dataset_entries=2)
        )


def test_file_dataset_flag_is_rejected_by_name(tmp_path: Path) -> None:
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

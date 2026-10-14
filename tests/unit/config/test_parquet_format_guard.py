# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A parquet input with a format that cannot read it fails with one clear error.

Config-time readers of format-specific files (dag root counting, tracelab and
weka sniffing) open input as text, so without an early guard a parquet file
dies in a raw ``UnicodeDecodeError`` before the composer can explain it.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.config import BenchmarkConfig
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

MESSAGE = "does not read parquet files; convert the data to JSONL"

NON_PARQUET_FORMATS = [
    param(CustomDatasetType.DAG_JSONL, id="dag_jsonl"),
    param(CustomDatasetType.TRACELAB, id="tracelab"),
    param(CustomDatasetType.WEKA_TRACE, id="weka_trace"),
]  # fmt: skip


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    path = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pylist([{"session_id": "a", "timestamp": 1}]), path)
    return path


@pytest.mark.parametrize("fmt", NON_PARQUET_FORMATS)
def test_cli_conversion_parquet_with_non_parquet_format_raises(
    parquet_path: Path, fmt: CustomDatasetType
) -> None:
    cli = CLIConfig(
        url="http://localhost:8000",
        model_names=["m"],
        input_file=parquet_path,
        custom_dataset_type=fmt,
    )
    with pytest.raises(ValueError, match=f"'{fmt}' {MESSAGE}"):
        convert_cli_to_aiperf(cli)


@pytest.mark.parametrize("fmt", NON_PARQUET_FORMATS)
def test_yaml_resolver_parquet_with_non_parquet_format_raises(
    parquet_path: Path, tmp_path: Path, fmt: CustomDatasetType
) -> None:
    config = BenchmarkConfig(
        models=["m"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "main",
                "type": "file",
                "path": str(parquet_path),
                "format": str(fmt),
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            }
        ],
    )
    run = BenchmarkRun(benchmark_id="t", cfg=config, artifact_dir=tmp_path / "out")
    with pytest.raises(ValueError, match=f"'{fmt}' {MESSAGE}"):
        DatasetResolver().resolve(run)


def test_cli_conversion_parquet_with_parquet_format_passes(
    parquet_path: Path,
) -> None:
    cli = CLIConfig(
        url="http://localhost:8000",
        model_names=["m"],
        input_file=parquet_path,
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
    )
    convert_cli_to_aiperf(cli)


DAG_ROW = {
    "session_id": "s",
    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
}


def _file_config(path: object, **dataset: object) -> BenchmarkConfig:
    return BenchmarkConfig(
        models=["m"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[{"name": "main", "type": "file", "path": path, **dataset}],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            }
        ],
    )


@pytest.fixture
def dag_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "dag.parquet"
    pq.write_table(pa.Table.from_pylist([DAG_ROW]), path)
    return path


def test_yaml_resolver_auto_detected_dag_parquet_raises(
    dag_parquet: Path, tmp_path: Path
) -> None:
    run = BenchmarkRun(
        benchmark_id="t", cfg=_file_config(str(dag_parquet)), artifact_dir=tmp_path
    )
    with pytest.raises(ValueError, match=f"'{CustomDatasetType.DAG_JSONL}' {MESSAGE}"):
        DatasetResolver().resolve(run)


def test_cli_auto_detected_dag_parquet_raises(dag_parquet: Path) -> None:
    run = make_run_from_cli(
        CLIConfig(
            url="http://localhost:8000", model_names=["m"], input_file=dag_parquet
        )
    )
    with pytest.raises(ValueError, match=f"'{CustomDatasetType.DAG_JSONL}' {MESSAGE}"):
        DatasetResolver().resolve(run)


@pytest.mark.parametrize("fmt", NON_PARQUET_FORMATS)
def test_yaml_resolver_remote_parquet_with_non_parquet_format_raises(
    dag_parquet: Path, tmp_path: Path, fmt: CustomDatasetType
) -> None:
    config = _file_config("https://example.com/dag.parquet", format=str(fmt))
    run = BenchmarkRun(benchmark_id="t", cfg=config, artifact_dir=tmp_path)
    with (
        patch(
            "aiperf.dataset.remote.service.cached_source",
            return_value=SimpleNamespace(local_path=dag_parquet),
        ),
        pytest.raises(ValueError, match=f"'{fmt}' {MESSAGE}"),
    ):
        DatasetResolver().resolve(run)

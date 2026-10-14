# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.dataset.composer.conftest import make_run

ROWS = [
    {"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0, 1]},
    {"timestamp": 5, "input_length": 12, "output_length": 3, "hash_ids": [0, 2]},
]
BASETEN_ROWS = [
    {
        "timestamp_start_unix_ms": 1,
        "prompt": "hi",
        "input_tokens": 3,
        "output_tokens": 2,
    }
]


def write_parquet(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _composer(path: Path, **kwargs) -> CustomDatasetComposer:
    cli = CLIConfig(model_names=["m"], input_file=str(path), **kwargs)
    return CustomDatasetComposer(run=make_run(cli), tokenizer=None)


def test_infer_dataset_type_mooncake_parquet_detects_mooncake(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    assert (
        _composer(path)._infer_dataset_type(str(path))
        == CustomDatasetType.MOONCAKE_TRACE
    )


def test_infer_dataset_type_mooncake_shard_directory_detects_mooncake(
    tmp_path: Path,
) -> None:
    write_parquet(tmp_path / "d" / "train-00000-of-00002.parquet", ROWS[:1])
    write_parquet(tmp_path / "d" / "train-00001-of-00002.parquet", ROWS[1:])
    composer = _composer(tmp_path / "d")
    assert (
        composer._infer_dataset_type(str(tmp_path / "d"))
        == CustomDatasetType.MOONCAKE_TRACE
    )


def test_infer_dataset_type_baseten_parquet_keeps_schema_priority(
    tmp_path: Path,
) -> None:
    path = write_parquet(tmp_path / "t.parquet", BASETEN_ROWS)
    assert (
        _composer(path)._infer_dataset_type(str(path))
        == CustomDatasetType.BASETEN_TRACE
    )


def test_create_dataset_parquet_with_non_parquet_format_raises(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    # The config converter rejects this pairing first, so hand the composer the
    # format directly to exercise its own guard.
    composer = _composer(path)
    with (
        patch.object(
            composer,
            "_explicit_format",
            return_value=CustomDatasetType.BURST_GPT_TRACE,
        ),
        patch.object(composer, "_create_loader_instance") as create_loader,
        pytest.raises(
            ValueError,
            match="Dataset format 'burst_gpt_trace' does not read parquet files",
        ),
    ):
        composer.create_dataset()
    create_loader.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        param({}, id="auto-detected"),
        param({"custom_dataset_type": CustomDatasetType.BASETEN_TRACE}, id="explicit"),
    ],
)  # fmt: skip
def test_create_dataset_baseten_shard_directory_raises(
    tmp_path: Path, kwargs: dict
) -> None:
    write_parquet(tmp_path / "d" / "train-00000-of-00002.parquet", BASETEN_ROWS)
    write_parquet(tmp_path / "d" / "train-00001-of-00002.parquet", BASETEN_ROWS)
    composer = _composer(tmp_path / "d", **kwargs)
    with (
        patch.object(composer, "_create_loader_instance") as create_loader,
        pytest.raises(
            ValueError, match="'baseten_trace' reads a single parquet or Arrow file"
        ),
    ):
        composer.create_dataset()
    create_loader.assert_not_called()

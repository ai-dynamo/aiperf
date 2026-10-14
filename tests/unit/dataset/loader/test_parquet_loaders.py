# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.base_trace_loader import _compute_file_hash
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType

ROWS = [
    {"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0, 1]},
    {"timestamp": 5, "input_length": 12, "output_length": 3, "hash_ids": [0, 2]},
]


def write_parquet(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class _DummyLoader(BaseFileLoader):
    def load_dataset(self):
        return {}

    def convert_to_conversations(self, custom_data):
        return []


def test_iter_record_dicts_parquet_file_yields_rows(tmp_path: Path) -> None:
    loader = _DummyLoader(filename=write_parquet(tmp_path / "t.parquet", ROWS))
    assert list(loader._iter_record_dicts()) == ROWS


def test_iter_record_dicts_shard_directory_yields_rows_in_order(tmp_path: Path) -> None:
    write_parquet(tmp_path / "d" / "train-00000-of-00002.parquet", ROWS[:1])
    write_parquet(tmp_path / "d" / "train-00001-of-00002.parquet", ROWS[1:])
    loader = _DummyLoader(filename=tmp_path / "d")
    assert list(loader._iter_record_dicts()) == ROWS


def test_compute_file_hash_single_parquet_hashes_file_bytes(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    assert (
        _compute_file_hash(str(path))
        == hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    )


def test_compute_file_hash_shard_directory_is_stable_when_moved(tmp_path: Path) -> None:
    write_parquet(tmp_path / "a" / "train-00000-of-00002.parquet", ROWS[:1])
    write_parquet(tmp_path / "a" / "train-00001-of-00002.parquet", ROWS[1:])
    moved = tmp_path / "elsewhere" / "b"
    shutil.copytree(tmp_path / "a", moved)
    assert _compute_file_hash(str(tmp_path / "a")) == _compute_file_hash(str(moved))
    assert (
        _compute_file_hash(str(tmp_path / "a"))
        != hashlib.sha256(str(tmp_path / "a").encode()).hexdigest()[:16]
    )


@pytest.mark.parametrize(
    "dataset_type,expected",
    [
        param(CustomDatasetType.MOONCAKE_TRACE, True, id="mooncake"),
        param(CustomDatasetType.BAILIAN_TRACE, True, id="bailian"),
        param(CustomDatasetType.SINGLE_TURN, True, id="single-turn"),
        param(CustomDatasetType.MULTI_TURN, True, id="multi-turn"),
        param(CustomDatasetType.BASETEN_TRACE, True, id="baseten"),
        param(CustomDatasetType.BURST_GPT_TRACE, False, id="burst-gpt"),
        param(CustomDatasetType.DAG_JSONL, False, id="dag-jsonl"),
        param(CustomDatasetType.RANDOM_POOL, False, id="random-pool"),
    ],
)  # fmt: skip
def test_reads_parquet_flag_matches_reader(
    dataset_type: CustomDatasetType, expected: bool
) -> None:
    LoaderClass = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, dataset_type)
    assert getattr(LoaderClass, "reads_parquet", False) is expected


def test_cli_config_accepts_parquet_shard_directory(tmp_path: Path) -> None:
    write_parquet(tmp_path / "d" / "train-00000-of-00001.parquet", ROWS)
    cli = CLIConfig(
        model_names=["m"],
        input_file=str(tmp_path / "d"),
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
    )
    assert str(cli.input_file) == str(tmp_path / "d")

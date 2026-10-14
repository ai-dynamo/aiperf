# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
import sys
from pathlib import Path
from unittest.mock import patch

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.common.record_files import (
    PYARROW_MISSING,
    ParquetReadError,
    ParquetSplitError,
    count_parquet_rows,
    first_record,
    is_parquet_input,
    iter_records,
    parquet_available,
    parquet_files,
    select_data_shards,
    select_parquet_shards,
    shard_splits,
)

ROWS = [
    {"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0, 1]},
    {"timestamp": 5, "input_length": 12, "output_length": 3, "hash_ids": [0, 2]},
]


def write_parquet(path: Path, rows: list[dict] | pa.Table) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def test_iter_records_jsonl_yields_each_line(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in ROWS) + b"\n\n")
    assert list(iter_records(path)) == ROWS


def test_iter_records_gzip_jsonl_yields_each_line(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl.gz"
    path.write_bytes(gzip.compress(b"\n".join(orjson.dumps(r) for r in ROWS)))
    assert list(iter_records(path)) == ROWS


def test_iter_records_invalid_jsonl_raises_value_error_with_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(
        ValueError, match=r"Invalid JSON in dataset file .*t\.jsonl at line 2"
    ):
        list(iter_records(path))


def test_iter_records_parquet_file_matches_jsonl_rows(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    assert list(iter_records(path)) == ROWS


def test_iter_records_drops_top_level_nulls(tmp_path: Path) -> None:
    rows = [{"timestamp": 1, "input_length": 4}, {"input_length": 5}]
    path = write_parquet(tmp_path / "t.parquet", rows)
    assert list(iter_records(path)) == rows


def test_iter_records_keeps_nested_lists_and_structs(tmp_path: Path) -> None:
    rows = [{"turns": [{"text": "hi", "delay": 1.5}], "ids": [[1, 2], [3]]}]
    path = write_parquet(tmp_path / "t.parquet", rows)
    assert list(iter_records(path)) == rows


def test_iter_records_dictionary_string_column_reads_as_str(tmp_path: Path) -> None:
    table = pa.table({"role": pa.array(["user", "user"]).dictionary_encode()})
    path = write_parquet(tmp_path / "t.parquet", table)
    assert list(iter_records(path)) == [{"role": "user"}, {"role": "user"}]


@pytest.mark.parametrize(
    "column,array",
    [
        param("image", pa.array([b"\x00"], type=pa.binary()), id="binary"),
        param("at", pa.array([0], type=pa.timestamp("ms")), id="timestamp"),
        param("price", pa.array([1], type=pa.decimal128(5, 2)), id="decimal"),
        param("m", pa.array([[("k", 1)]], type=pa.map_(pa.string(), pa.int64())), id="map"),
    ],
)  # fmt: skip
def test_iter_records_unsupported_column_type_raises_named_error(
    tmp_path: Path, column: str, array: pa.Array
) -> None:
    path = write_parquet(tmp_path / "t.parquet", pa.table({column: array}))
    with pytest.raises(
        ParquetReadError, match=rf"parquet column '{column}' in t\.parquet has type"
    ):
        list(iter_records(path))


def test_iter_records_non_finite_float_raises(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "t.parquet", [{"delay": 1.0}, {"delay": float("nan")}]
    )
    with pytest.raises(ParquetReadError, match="'delay'.*NaN or infinity"):
        list(iter_records(path))


def test_iter_records_corrupt_parquet_raises_named_error(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    path.write_bytes(b'{"timestamp": 0}\n')
    with pytest.raises(
        ParquetReadError, match=r"t\.parquet is not a readable parquet file"
    ):
        list(iter_records(path))


def test_iter_records_shard_directory_reads_in_shard_order(tmp_path: Path) -> None:
    write_parquet(tmp_path / "d" / "train-00001-of-00002.parquet", ROWS[1:])
    write_parquet(tmp_path / "d" / "train-00000-of-00002.parquet", ROWS[:1])
    assert list(iter_records(tmp_path / "d")) == ROWS


def test_is_parquet_input_ignores_docs_and_manifests(tmp_path: Path) -> None:
    d = tmp_path / "d"
    write_parquet(d / "0000.parquet", ROWS)
    for name in (
        ".gitattributes",
        "README.md",
        "aiperf.json",
        "0000.parquet.aiperf.json",
    ):
        (d / name).write_text("{}")
    assert is_parquet_input(d)
    assert parquet_files(d) == [d / "0000.parquet"]


@pytest.mark.parametrize(
    "files,expected",
    [
        param(["t.parquet"], True, id="parquet-file"),
        param(["t.jsonl"], False, id="jsonl-file"),
        param(["d/0000.parquet", "d/0001.parquet"], True, id="shard-dir"),
        param(["d/0000.parquet", "d/extra.jsonl"], False, id="mixed-dir"),
        param(["d/README.md"], False, id="docs-only-dir"),
    ],
)  # fmt: skip
def test_is_parquet_input_classifies_paths(
    tmp_path: Path, files: list[str], expected: bool
) -> None:
    for rel in files:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".parquet"):
            write_parquet(target, ROWS)
        else:
            target.write_text("{}\n")
    path = tmp_path / ("d" if files[0].startswith("d/") else files[0])
    assert is_parquet_input(path) is expected


def test_is_parquet_input_missing_path_is_false(tmp_path: Path) -> None:
    assert not is_parquet_input(tmp_path / "nope.parquet")


@pytest.mark.parametrize(
    "names,expected",
    [
        param(["0001.parquet", "0000.parquet"], ["0000.parquet", "0001.parquet"], id="unnamed-sorted"),
        param(
            ["test-00000-of-00001.parquet", "train-00001-of-00002.parquet", "train-00000-of-00002.parquet"],
            ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
            id="train-chosen",
        ),
        param(["validation-00000-of-00001.parquet"], ["validation-00000-of-00001.parquet"], id="single-split"),
        param(["README.md", "a.parquet"], ["a.parquet"], id="non-parquet-dropped"),
    ],
)  # fmt: skip
def test_select_parquet_shards_picks_read_order(
    names: list[str], expected: list[str]
) -> None:
    assert select_parquet_shards(names) == expected


def test_shard_splits_lists_splits_of_parquet_names_only() -> None:
    names = ["test-00000-of-00001.parquet", "train-00000-of-00002.parquet",
             "train-00001-of-00002.parquet", "0000.parquet", "README.md"]  # fmt: skip
    assert shard_splits(names) == ["", "test", "train"]


def test_select_parquet_shards_no_train_raises_with_splits() -> None:
    names = ["test-00000-of-00001.parquet", "validation-00000-of-00001.parquet"]
    with pytest.raises(ParquetSplitError, match="test, validation") as exc_info:
        select_parquet_shards(names)
    assert exc_info.value.splits == {
        "test": ["test-00000-of-00001.parquet"],
        "validation": ["validation-00000-of-00001.parquet"],
    }


@pytest.mark.parametrize(
    "names,expected",
    [
        param(["README.md", ".gitattributes", "b.parquet", "a.parquet"], ["a.parquet", "b.parquet"], id="docs-ignored"),
        param(["a.parquet", "b.jsonl"], None, id="mixed"),
        param(["README.md"], None, id="docs-only"),
        param([], None, id="empty"),
    ],
)  # fmt: skip
def test_select_data_shards_returns_shards_or_none(
    names: list[str], expected: list[str] | None
) -> None:
    assert select_data_shards(names) == expected


def test_select_data_shards_no_train_raises() -> None:
    with pytest.raises(ParquetSplitError):
        select_data_shards(
            ["test-00000-of-00001.parquet", "validation-00000-of-00001.parquet"]
        )


def test_first_record_returns_first_row(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    assert first_record(path) == ROWS[0]


def test_first_record_empty_parquet_returns_none(tmp_path: Path) -> None:
    table = pa.table({"timestamp": pa.array([], type=pa.int64())})
    path = write_parquet(tmp_path / "t.parquet", table)
    assert first_record(path) is None
    assert list(iter_records(path)) == []


def test_count_parquet_rows_sums_shards(tmp_path: Path) -> None:
    write_parquet(tmp_path / "d" / "train-00000-of-00002.parquet", ROWS)
    write_parquet(tmp_path / "d" / "train-00001-of-00002.parquet", ROWS[:1])
    assert count_parquet_rows(tmp_path / "d") == 3


def test_iter_records_without_pyarrow_raises_dependency_message(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "t.parquet", ROWS)
    with (
        patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}),
        pytest.raises(ParquetReadError) as exc_info,
    ):
        list(iter_records(path))
    assert str(exc_info.value) == PYARROW_MISSING
    with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
        assert not parquet_available()

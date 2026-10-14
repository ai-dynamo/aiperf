# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same records as JSONL and as parquet are indistinguishable to every reader."""

from pathlib import Path
from unittest.mock import patch

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.common.record_files import iter_records
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags._converter_profiling import _count_dataset_records
from aiperf.dataset.trace_timestamps import first_record_has_timestamp
from aiperf.plugin.enums import CustomDatasetType

ROWS = [
    {"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0, 1]},
    {"timestamp": 5, "input_length": 12, "output_length": 3, "hash_ids": [0, 2]},
    {"timestamp": 9, "input_length": 8, "output_length": 4, "hash_ids": [3]},
]
UNTIMED = [{k: v for k, v in r.items() if k != "timestamp"} for r in ROWS]
SESSIONS = [
    {"session_id": "a", "turns": [{"text": "hi"}]},
    {"session_id": "a", "turns": [{"text": "again"}]},
    {"session_id": "b", "turns": [{"text": "yo"}]},
]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")
    return path


def _parquet(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _shards(directory: Path, rows: list[dict]) -> Path:
    _parquet(directory / "train-00000-of-00002.parquet", rows[:1])
    _parquet(directory / "train-00001-of-00002.parquet", rows[1:])
    return directory


@pytest.fixture(params=["file", "shards"])
def encodings(request, tmp_path: Path):
    def make(rows: list[dict]) -> tuple[Path, Path]:
        jsonl = _jsonl(tmp_path / "t.jsonl", rows)
        if request.param == "file":
            return jsonl, _parquet(tmp_path / "t.parquet", rows)
        return jsonl, _shards(tmp_path / "d", rows)

    return make


def test_iter_records_parquet_equals_jsonl(encodings) -> None:
    jsonl, parquet = encodings(ROWS)
    assert list(iter_records(parquet)) == list(iter_records(jsonl))


def test_detect_type_parquet_equals_jsonl(encodings) -> None:
    jsonl, parquet = encodings(ROWS)
    detected, record = DatasetResolver._detect_type(str(parquet))
    assert (
        detected
        == DatasetResolver._detect_type(str(jsonl))[0]
        == CustomDatasetType.MOONCAKE_TRACE
    )
    assert record == ROWS[0]


@pytest.mark.parametrize(
    "rows,expected",
    [
        param(ROWS, True, id="timed"),
        param(UNTIMED, False, id="untimed"),
    ],
)  # fmt: skip
def test_timestamp_detection_parquet_equals_jsonl(
    encodings, rows: list[dict], expected: bool
) -> None:
    jsonl, parquet = encodings(rows)
    assert (
        first_record_has_timestamp(parquet)
        is first_record_has_timestamp(jsonl)
        is expected
    )
    assert (
        DatasetResolver._check_timing_data(str(parquet), None)
        is DatasetResolver._check_timing_data(str(jsonl), None)
        is expected
    )


def test_record_counts_parquet_equal_jsonl(encodings) -> None:
    jsonl, parquet = encodings(ROWS)
    assert _count_dataset_records(parquet) == _count_dataset_records(jsonl) == 3
    assert DatasetResolver._count_records_and_sessions(
        str(parquet), CustomDatasetType.MOONCAKE_TRACE
    ) == DatasetResolver._count_records_and_sessions(
        str(jsonl), CustomDatasetType.MOONCAKE_TRACE
    )


def test_session_counts_parquet_equal_jsonl(encodings) -> None:
    jsonl, parquet = encodings(SESSIONS)
    expected = (3, 2)
    assert (
        DatasetResolver._count_records_and_sessions(
            str(jsonl), CustomDatasetType.MULTI_TURN
        )
        == expected
    )
    assert (
        DatasetResolver._count_records_and_sessions(
            str(parquet), CustomDatasetType.MULTI_TURN
        )
        == expected
    )


def test_timestamp_detection_baseten_parquet_still_true(tmp_path: Path) -> None:
    path = _parquet(
        tmp_path / "t.parquet", [{"timestamp_start_unix_ms": 1, "prompt": "p"}]
    )
    assert first_record_has_timestamp(path)


def _fail(*args: object, **kwargs: object) -> None:
    raise AssertionError("parquet rows must not be decoded here")


def test_detect_type_baseten_parquet_reads_schema_only(tmp_path: Path) -> None:
    path = _parquet(
        tmp_path / "t.parquet",
        [{"timestamp_start_unix_ms": 1, "prompt": "p", "input_tokens": 3, "output_tokens": 2}],
    )  # fmt: skip
    with patch("aiperf.common.record_files.first_record", _fail):
        detected, record = DatasetResolver._detect_type(str(path))
    assert detected == CustomDatasetType.BASETEN_TRACE
    assert record is None


def test_record_counts_parquet_single_turn_read_metadata_only(encodings) -> None:
    _, parquet = encodings(ROWS)
    with patch("aiperf.common.record_files.iter_records", _fail):
        assert DatasetResolver._count_records_and_sessions(
            str(parquet), CustomDatasetType.MOONCAKE_TRACE
        ) == (3, 3)

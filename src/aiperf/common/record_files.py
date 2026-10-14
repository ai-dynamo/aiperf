# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read dataset records from JSONL or parquet input.

A parquet row becomes the same dict the JSONL path produces from one line, so
every loader that reads through :func:`iter_records` accepts both. pyarrow is
imported lazily: it has no Windows-on-ARM wheel.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.utils import open_text_maybe_gzip

PARQUET_SUFFIX = ".parquet"
DEFAULT_SPLIT = "train"
PYARROW_MISSING = (
    "parquet datasets need pyarrow, which has no Windows-on-ARM wheel "
    "(apache/arrow#47195)"
)
# pyarrow's own default; one batch is decoded into Python objects at a time.
_BATCH_ROWS = 65_536
_HF_SHARD_NAME = re.compile(r"^(?P<split>.+)-\d{5}-of-\d{5}\.parquet$", re.IGNORECASE)


class ParquetReadError(ValueError):
    """Parquet input cannot be read as AIPerf records."""


class ParquetSplitError(ParquetReadError):
    """Parquet shards hold several splits and none is the default one."""

    def __init__(self, splits: dict[str, list[str]]) -> None:
        self.splits = splits
        names = ", ".join(sorted(name or "(unnamed)" for name in splits))
        super().__init__(
            f"the parquet files hold the splits {names} and none is named "
            f"'{DEFAULT_SPLIT}'; point at one split's files"
        )


def is_ignored_name(name: str) -> bool:
    """Whether a file beside parquet shards is documentation or metadata, not data."""
    lowered = name.lower()
    return (
        name.startswith(".")
        or lowered.startswith("readme")
        or lowered == "aiperf.json"
        or lowered.endswith(".aiperf.json")
    )


def _split_groups(names: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in sorted(n for n in names if n.lower().endswith(PARQUET_SUFFIX)):
        match = _HF_SHARD_NAME.match(name)
        groups.setdefault(match.group("split") if match else "", []).append(name)
    return groups


def shard_splits(names: Sequence[str]) -> list[str]:
    """The sorted split names among the parquet ``names``; '' for files outside split naming."""
    return sorted(_split_groups(names))


def select_parquet_shards(names: Sequence[str]) -> list[str]:
    """The parquet file names to read from one directory, in read order.

    Names following Hugging Face's ``<split>-NNNNN-of-NNNNN.parquet`` are grouped
    by split; with more than one split, ``train`` is read. Shard numbers are
    zero-padded, so sorted order is publication order.

    Raises:
        ParquetSplitError: Several splits are present and none is ``train``.
    """
    groups = _split_groups(names)
    if len(groups) <= 1:
        return next(iter(groups.values()), [])
    if DEFAULT_SPLIT in groups:
        return groups[DEFAULT_SPLIT]
    raise ParquetSplitError(groups)


def select_data_shards(names: Sequence[str]) -> list[str] | None:
    """The shards to read from one folder's file names, or None when it is not parquet input.

    Documentation and manifest files are ignored; any other non-parquet file means
    the folder is not a parquet dataset.

    Raises:
        ParquetSplitError: Several splits are present and none is ``train``.
    """
    data = [name for name in names if not is_ignored_name(name)]
    if not data or not all(n.lower().endswith(PARQUET_SUFFIX) for n in data):
        return None
    return select_parquet_shards(data)


def _data_names(directory: Path) -> list[str]:
    return [
        child.name
        for child in directory.iterdir()
        if child.is_file() and not is_ignored_name(child.name)
    ]


def is_parquet_input(path: str | Path) -> bool:
    """Whether ``path`` is a parquet file, or a directory whose data files are all parquet."""
    p = Path(path)
    if p.is_file():
        return p.suffix.lower() == PARQUET_SUFFIX
    if not p.is_dir():
        return False
    names = _data_names(p)
    return bool(names) and all(n.lower().endswith(PARQUET_SUFFIX) for n in names)


def parquet_files(path: str | Path) -> list[Path]:
    """The parquet files to read for an :func:`is_parquet_input` path, in read order.

    Raises:
        ParquetSplitError: The directory holds several splits and none is ``train``.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    return [p / name for name in select_parquet_shards(_data_names(p))]


def require_parquet_reader(dataset_type: object) -> None:
    """Reject a dataset format whose loader reads files itself and so cannot take parquet input.

    Call it only for parquet input; every caller guards with :func:`is_parquet_input`.

    Raises:
        ValueError: The loader for ``dataset_type`` does not set ``reads_parquet``.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    LoaderClass = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, dataset_type)
    if not getattr(LoaderClass, "reads_parquet", False):
        raise ValueError(
            f"Dataset format '{dataset_type}' does not read parquet files; "
            "convert the data to JSONL or choose a format that reads parquet"
        )


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ParquetReadError(PYARROW_MISSING) from e
    return pa, pq


def parquet_available() -> bool:
    """Whether pyarrow can be imported on this platform."""
    try:
        _pyarrow()
    except ParquetReadError:
        return False
    return True


def _supported(pa: Any, arrow_type: Any) -> bool:
    types = pa.types
    if types.is_dictionary(arrow_type):
        return types.is_string(arrow_type.value_type) or types.is_large_string(
            arrow_type.value_type
        )
    if (
        types.is_list(arrow_type)
        or types.is_large_list(arrow_type)
        or types.is_fixed_size_list(arrow_type)
    ):
        return _supported(pa, arrow_type.value_type)
    if types.is_struct(arrow_type):
        return all(
            _supported(pa, arrow_type.field(i).type)
            for i in range(arrow_type.num_fields)
        )
    return (
        types.is_integer(arrow_type)
        or types.is_floating(arrow_type)
        or types.is_boolean(arrow_type)
        or types.is_string(arrow_type)
        or types.is_large_string(arrow_type)
        or types.is_null(arrow_type)
    )


def _has_floating(pa: Any, arrow_type: Any) -> bool:
    types = pa.types
    if types.is_floating(arrow_type):
        return True
    if (
        types.is_list(arrow_type)
        or types.is_large_list(arrow_type)
        or types.is_fixed_size_list(arrow_type)
    ):
        return _has_floating(pa, arrow_type.value_type)
    if types.is_struct(arrow_type):
        return any(
            _has_floating(pa, arrow_type.field(i).type)
            for i in range(arrow_type.num_fields)
        )
    return False


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return True


def _check_finite(row: dict[str, Any], float_columns: list[str], file: Path) -> None:
    for name in float_columns:
        if not _all_finite(row.get(name)):
            raise ParquetReadError(
                f"parquet column '{name}' in {file.name} holds NaN or "
                "infinity, which AIPerf records cannot hold"
            )


def _iter_parquet_file(
    pa: Any, pq: Any, file: Path, batch_rows: int
) -> Iterator[dict[str, Any]]:
    try:
        parquet_file = pq.ParquetFile(file)
    except (OSError, pa.ArrowException) as e:
        raise ParquetReadError(
            f"{file.name} is not a readable parquet file: {e}"
        ) from e
    with closing(parquet_file):
        schema = parquet_file.schema_arrow
        for field in schema:
            if not _supported(pa, field.type):
                raise ParquetReadError(
                    f"parquet column '{field.name}' in {file.name} has type "
                    f"{field.type}, which AIPerf records cannot hold"
                )
        float_columns = [f.name for f in schema if _has_floating(pa, f.type)]
        for batch in parquet_file.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                _check_finite(row, float_columns, file)
                # push_to_hub turns a key a row never had into null; dropping
                # nulls restores the record shape the JSONL file had.
                yield {key: value for key, value in row.items() if value is not None}


def _iter_parquet(
    files: Sequence[Path], batch_rows: int = _BATCH_ROWS
) -> Iterator[dict[str, Any]]:
    pa, pq = _pyarrow()
    for file in files:
        yield from _iter_parquet_file(pa, pq, file, batch_rows)


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every record of JSONL (optionally gzipped) or parquet input as a dict.

    Raises:
        ValueError: A JSONL line is not valid JSON.
        ParquetReadError: pyarrow is missing, a file is not parquet, a column type
            or value has no JSON equivalent, or the split to read is ambiguous.
    """
    if is_parquet_input(path):
        yield from _iter_parquet(parquet_files(path))
        return
    with open_text_maybe_gzip(path) as f:
        for lineno, line in enumerate(f, start=1):
            if line := line.strip():
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in dataset file {path} at line {lineno}: {e}"
                    ) from None


def first_record(path: str | Path) -> dict[str, Any] | None:
    """The first record of ``path``, or None when it has none. Raises like :func:`iter_records`."""
    # One-row batches: a default batch would decode 64K rows to read one.
    records = (
        _iter_parquet(parquet_files(path), batch_rows=1)
        if is_parquet_input(path)
        else iter_records(path)
    )
    try:
        return next(records, None)
    finally:
        records.close()


def count_parquet_rows(path: str | Path) -> int:
    """Rows across the parquet files of ``path``, read from file metadata only.

    Raises:
        ParquetReadError: pyarrow is missing, or the split to read is ambiguous.
    """
    _, pq = _pyarrow()
    total = 0
    for file in parquet_files(path):
        with closing(pq.ParquetFile(file)) as parquet_file:
            total += parquet_file.metadata.num_rows
    return total

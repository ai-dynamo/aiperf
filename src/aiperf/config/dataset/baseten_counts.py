# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseten trace dataset counting helpers."""


def count_baseten_parquet_records_and_sessions(file_path: str) -> tuple[int, int]:
    """Return row and session counts for a Baseten Parquet trace file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return 0, 0

    try:
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        schema_names = set(pq.read_schema(file_path).names)
        session_columns = [
            column
            for column in ("provided_session_id", "poor_man_session_id")
            if column in schema_names
        ]
        if not session_columns:
            return row_count, row_count

        table = pq.read_table(file_path, columns=session_columns)
    except (OSError, pa.ArrowException):
        return 0, 0

    for column in session_columns:
        values = {
            str(row[column]) for row in table.to_pylist() if row.get(column) is not None
        }
        if values:
            return row_count, len(values)
    return row_count, row_count

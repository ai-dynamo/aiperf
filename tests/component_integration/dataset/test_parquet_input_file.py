# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A local directory of parquet shards replays every row of every shard, in order."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

_ROWS = [
    {"input_length": 48, "output_length": 8, "hash_ids": [1, 2, 3], "timestamp": 100},
    {"input_length": 48, "output_length": 8, "hash_ids": [4, 5, 6], "timestamp": 200},
    {"input_length": 48, "output_length": 8, "hash_ids": [7, 8, 9], "timestamp": 300},
]


@pytest.mark.component_integration
def test_parquet_shard_directory_replays_every_row(
    cli: AIPerfCLI, tmp_path: Path
) -> None:
    shards = tmp_path / "mooncake"
    shards.mkdir()
    pq.write_table(
        pa.Table.from_pylist(_ROWS[:2]), shards / "train-00000-of-00002.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(_ROWS[2:]), shards / "train-00001-of-00002.parquet"
    )
    result = cli.run_sync(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --endpoint-type chat \
            --custom-dataset-type mooncake_trace \
            --input-file {shards} \
            --isl-block-size 16 \
            --workers-max {defaults.workers_max} \
            --export-level raw \
            --ui {defaults.ui}
        """,
        timeout=90.0,
    )
    assert result.raw_records is not None
    assert len(result.raw_records) == len(_ROWS)

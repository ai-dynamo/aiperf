# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration coverage for Baseten Parquet trace replay."""

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.mark.component_integration
class TestBasetenTraceReplay:
    def test_baseten_trace_cli_forwards_replay_metadata(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        input_file = _write_parquet(
            tmp_path / "baseten.parquet",
            [
                {
                    "timestamp_start_unix_ms": 100,
                    "prompt": "first prompt",
                    "input_tokens": 3,
                    "output_tokens": 6,
                    "total_hashes": [1, 2],
                    "provided_session_id": "unique-1",
                    "poor_man_session_id": 7,
                    "block_size": 64,
                },
                {
                    "timestamp_start_unix_ms": 200,
                    "prompt": "second prompt",
                    "input_tokens": 4,
                    "output_tokens": 8,
                    "total_hashes": [3, 4],
                    "provided_session_id": "unique-2",
                    "poor_man_session_id": 7,
                    "block_size": 64,
                },
            ],
        )

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type completions \
                --custom-dataset-type baseten_trace \
                --input-file {input_file} \
                --fixed-schedule \
                --trace-session-sample-ratio 1.0 \
                --num-conversations 1 \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == 2

        payloads = [record.payload for record in result.raw_records]
        # On-wire invariant: a single recorded prompt is sent as the exact bare
        # string (canonical OpenAI form; Baseten's /v1/completions gateway
        # rejects list[str]). CompletionsEndpoint emits the string natively, so
        # this holds whether or not the loader also carries the prompt through
        # extra_body.
        assert [payload["prompt"] for payload in payloads] == [
            "first prompt",
            "second prompt",
        ]
        assert payloads[0]["max_tokens"] == 6
        assert payloads[0]["min_tokens"] == 6
        assert payloads[0]["hash_ids"] == [1, 2]
        assert payloads[0]["block_size"] == 64

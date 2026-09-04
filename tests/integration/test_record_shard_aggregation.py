# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for per-RecordProcessor shard export + aggregation.

Runs a full `aiperf profile` against the mock server with multiple
RecordProcessor services so the computed-record JSONL export fans out into
per-processor `records_shards/records_{id}.jsonl` shards, then asserts the
`data_exporter` aggregator merged every shard into the final
`profile_export.jsonl` (and cleaned the shard directory up).
"""

from __future__ import annotations

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults

# Enough RecordProcessors to guarantee the export fans out across >1 shard.
_RECORD_PROCESSORS = 3


def _shard_dirs(results, name: str) -> list:
    """Any surviving shard subdirectories named `name` under the artifact dir."""
    return [p for p in results.artifacts_dir.glob(f"**/{name}") if p.is_dir()]


def _count_shard_writers(results) -> int:
    """How many per-processor record shard writers announced themselves in the log."""
    return results.log.count("Record shard JSONL writer enabled")


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecordShardAggregation:
    """Full-run validation of computed-record sharding + aggregation."""

    async def test_records_level_aggregates_all_shards(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """--export-level records: every per-processor shard merges into profile_export.jsonl."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --record-processors {_RECORD_PROCESSORS} \
                --export-level records \
                --ui none
            """
        )

        assert result.exit_code == 0

        # The export fanned out across multiple per-processor shard writers...
        assert _count_shard_writers(result) == _RECORD_PROCESSORS

        # ...and the aggregator merged EVERY record into the single final file.
        assert result.jsonl is not None
        assert len(result.jsonl) == defaults.request_count
        # Every merged line is a well-formed record (harness parses as MetricRecordInfo).
        assert all(r.metrics or r.error for r in result.jsonl)

        # The transient shard directory was consumed and removed post-aggregation.
        assert _shard_dirs(result, "records_shards") == []

    async def test_raw_level_aggregates_records_and_raw_shards(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """--export-level raw: both computed-record and raw shards aggregate + clean up."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --record-processors {_RECORD_PROCESSORS} \
                --export-level raw \
                --ui none
            """
        )

        assert result.exit_code == 0

        # Computed-record shards merged into profile_export.jsonl.
        assert result.jsonl is not None
        assert len(result.jsonl) == defaults.request_count

        # Raw-record shards merged into profile_export_raw.jsonl.
        assert result.raw_records is not None
        assert len(result.raw_records) == defaults.request_count

        # Both transient shard directories were removed post-aggregation.
        assert _shard_dirs(result, "records_shards") == []
        assert _shard_dirs(result, "raw_records") == []

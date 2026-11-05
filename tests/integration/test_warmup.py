# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for warmup phase functionality."""

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestWarmup:
    """Tests for warmup phase functionality."""

    async def test_warmup_phase(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Warmup requests excluded from profiling metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 15 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 15

    async def test_disjoint_warmup_exports_actual_phase_membership(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """Real CLI, HTTP requests, and export must agree after both pools wrap."""
        result = await cli.run(
            f"""
            aiperf profile --model {defaults.model}
                --url {aiperf_mock_server.url} --endpoint-type chat
                --num-dataset-entries 6 --warmup-dataset-entries 2
                --dataset-sampling-strategy shuffle --random-seed 42
                --warmup-request-count 5 --request-count 9
                --isl 16 --osl 4 --concurrency 1 --workers-max 1
                --ui {defaults.ui} --export-outputs-json
            """
        )
        outputs = orjson.loads(
            next(result.artifacts_dir.rglob("outputs.json")).read_bytes()
        )
        warmup = outputs["warmup"]
        profiling = outputs["data"]
        assert len(warmup) == 5
        assert len(profiling) == result.request_count == 9
        warmup_ids = {record["conversation_id"] for record in warmup}
        profiling_ids = {record["conversation_id"] for record in profiling}
        assert len(warmup_ids) == 2
        assert len(profiling_ids) == 4
        assert warmup_ids.isdisjoint(profiling_ids)

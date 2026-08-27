# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestDeterministicBehavior:
    """Tests for deterministic behavior with random seeds."""

    async def test_same_seed_identical_inputs(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Same random seed produces identical payloads across runs."""

        # Note: Using multiple workers to ensure that the random seed is reproducible.
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max 5 \
                --ui {defaults.ui}
            """
        )

        result2 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max 5 \
                --ui {defaults.ui}
            """
        )

        assert result1.request_count == defaults.request_count
        assert result2.request_count == defaults.request_count

        inputs_1 = result1.inputs.data
        inputs_2 = result2.inputs.data

        assert len(inputs_1) == len(inputs_2), "Session counts differ"

        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            assert s1.payloads == s2.payloads

    async def test_different_seeds_different_inputs(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Different random seeds produce different payloads."""
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        result2 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency 2 \
                --random-seed 123 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result1.request_count == defaults.request_count
        assert result2.request_count == defaults.request_count

        inputs_1 = result1.inputs.data
        inputs_2 = result2.inputs.data

        payloads_different = False
        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            if s1.payloads != s2.payloads:
                payloads_different = True
                break

        assert payloads_different, "Different seeds should produce different payloads"

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic behavior and random seed tests."""

import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestDeterministicBehavior:
    """Tests for deterministic behavior with random seeds."""

    async def test_same_seed_identical_inputs(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Same random seed produces identical payloads."""
        # Run first benchmark with seed 42
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )

        # Run second benchmark with same seed
        result2 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --ui simple
            """
        )

        # Both should complete same number of requests
        assert result1.request_count == 10
        assert result2.request_count == 10

        # Load inputs data
        inputs_1 = result1._inputs.get("data", [])
        inputs_2 = result2._inputs.get("data", [])

        assert len(inputs_1) == len(inputs_2), "Session counts differ"

        # Verify payloads are identical (session IDs will differ as they're UUIDs)
        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            # Session IDs should differ (they're UUIDs)
            assert s1.get("session_id") != s2.get("session_id")
            # But payloads should be identical
            assert s1.get("payloads") == s2.get("payloads")

    async def test_different_seeds_different_inputs(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Different random seeds produce different payloads."""
        # Run with seed 42
        result1 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """
        )

        # Run with seed 123
        result2 = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed 123 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --ui simple
            """
        )

        # Both should complete
        assert result1.request_count == 10
        assert result2.request_count == 10

        # Load inputs data
        inputs_1 = result1._inputs.get("data", [])
        inputs_2 = result2._inputs.get("data", [])

        # Payloads should be different
        payloads_different = False
        for s1, s2 in zip(inputs_1, inputs_2, strict=True):
            if s1.get("payloads") != s2.get("payloads"):
                payloads_different = True
                break

        assert payloads_different, "Different seeds should produce different payloads"

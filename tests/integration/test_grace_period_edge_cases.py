# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for grace period edge cases."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestGracePeriodEdgeCases:
    """Tests for grace period behavior in various scenarios."""

    async def test_grace_period_completes_multi_turn_conversations(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test grace period allows multi-turn conversations to complete all turns.

        When duration expires, in-flight multi-turn conversations should have time
        to complete all their turns within the grace period.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 10 \
                --random-seed 42 \
                --request-rate-mode constant \
                --conversation-num 15 \
                --conversation-turn-mean 3 \
                --conversation-turn-stddev 0 \
                --benchmark-duration 2 \
                --benchmark-grace-period 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # With 15 3-turn conversations and sufficient grace period,
        # conversations that started should complete all their turns
        assert result.request_count >= 11

        # Group records by conversation to verify turn completion
        conversation_turns: dict[str, set[int]] = {}
        for record in result.jsonl:
            conv_id = record.metadata.conversation_id
            turn = record.metadata.turn_index
            if conv_id not in conversation_turns:
                conversation_turns[conv_id] = set()
            conversation_turns[conv_id].add(turn)

        # Conversations that started (have turn 0) should complete all 3 turns
        # due to sufficient grace period
        complete_conversations = sum(
            1 for turns in conversation_turns.values() if turns == {0, 1, 2}
        )
        started_conversations = sum(
            1 for turns in conversation_turns.values() if 0 in turns
        )
        # Most started conversations should complete (allow some variance for timing)
        assert complete_conversations >= started_conversations - 2

    async def test_duration_limits_new_requests(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that duration stops issuing new requests regardless of grace period.

        Even with a long grace period, only requests issued within the duration
        window should run. Grace period only affects in-flight request completion.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-rate 10 \
                --request-count 50 \
                --random-seed 42 \
                --request-rate-mode constant \
                --benchmark-duration 1 \
                --benchmark-grace-period 60 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # With rate=10 and duration=1s, ~10 requests should be issued
        # Grace period allows completion but doesn't issue new requests
        assert result.request_count <= 12

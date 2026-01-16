# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for dataset sampling strategies.

Tests the --dataset-sampling-strategy parameter which controls how requests
sample from the generated dataset: sequential, shuffle, random.
"""

from collections import Counter

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.timing.conftest import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestDatasetSamplingStrategies:
    """Tests for dataset sampling strategies."""

    def test_sequential_sampling(self, cli: AIPerfCLI):
        """Test sequential dataset sampling (default behavior)."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy sequential \
                --random-seed 42 \
                --num-sessions 10 \
                --session-turns-mean 1 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Verify request count
        assert len(result.jsonl) == 10

        # Verify credit flow
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.credits_balanced()
        assert analyzer.total_credits == 10

        # Validate sequential order
        conversation_ids = [r.metadata.conversation_id for r in result.jsonl]
        assert conversation_ids == [f"session_{i:06d}" for i in range(10)], (
            f"Sequential sampling should produce IDs in order [0-9], got {conversation_ids}"
        )

    def test_shuffle_sampling(self, cli: AIPerfCLI):
        """Test shuffle dataset sampling produces shuffled order with unique IDs."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy shuffle \
                --random-seed 42 \
                --num-sessions 15 \
                --num-dataset-entries 15 \
                --session-turns-mean 1 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Verify request count
        assert len(result.jsonl) == 15

        # Verify credit flow
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.credits_balanced()
        assert analyzer.total_credits == 15

        # Validate shuffle produces unique IDs (no repeats in single pass)
        conversation_ids = [r.metadata.conversation_id for r in result.jsonl]

        # Should have all unique IDs
        assert len(set(conversation_ids)) == 15, (
            f"Shuffle should produce 15 unique IDs, got {len(set(conversation_ids))}"
        )

        # Should differ from sequential order (shuffle actually shuffles)
        sequential_order = [f"session_{i:06d}" for i in range(15)]
        assert conversation_ids != sequential_order, (
            "Shuffle should produce different order than sequential [0-14]"
        )

    def test_random_sampling(self, cli: AIPerfCLI):
        """Test random dataset sampling with replacement."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy random \
                --random-seed 42 \
                --num-sessions 12 \
                --session-turns-mean 1 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Verify request count
        assert len(result.jsonl) == 12

        # Verify credit flow
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.credits_balanced()

        # Validate random sampling with replacement
        conversation_ids = [r.metadata.conversation_id for r in result.jsonl]
        counts = Counter(conversation_ids)

        # With replacement: expect some IDs to repeat
        max_count = max(counts.values())
        assert max_count > 1, (
            f"Random with replacement should have duplicates, all counts: {dict(counts)}"
        )

    def test_sequential_wrapping(self, cli: AIPerfCLI):
        """Test sequential sampling wraps around when requests exceed dataset size."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy sequential \
                --random-seed 42 \
                --num-dataset-entries 5 \
                --num-sessions 15 \
                --session-turns-mean 1 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Should complete all 15 requests by wrapping around the 5-entry dataset
        assert len(result.jsonl) == 15

        # Extract session IDs (should see same sessions repeated)
        session_ids = [record.metadata.conversation_id for record in result.jsonl]

        # With 5 dataset entries and 15 requests, each entry should be used 3 times
        session_counts = Counter(session_ids)
        assert len(session_counts) == 5, "Should have exactly 5 unique sessions"

        # Each session should appear exactly 3 times (15 / 5 = 3)
        for session_id, count in session_counts.items():
            assert count == 3, (
                f"Session {session_id} appeared {count} times, expected 3"
            )

    def test_shuffle_resampling(self, cli: AIPerfCLI):
        """Test shuffle strategy reshuffles after exhausting dataset."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy shuffle \
                --random-seed 42 \
                --num-dataset-entries 5 \
                --num-sessions 15 \
                --session-turns-mean 1 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Should complete all 15 requests by reshuffling the 5-entry dataset
        assert len(result.jsonl) == 15

        # Extract session IDs
        session_ids = [record.metadata.conversation_id for record in result.jsonl]

        # With 5 dataset entries and 15 requests, each entry should be used 3 times
        session_counts = Counter(session_ids)
        assert len(session_counts) == 5, "Should have exactly 5 unique sessions"

        # Each session should appear exactly 3 times
        for session_id, count in session_counts.items():
            assert count == 3, (
                f"Session {session_id} appeared {count} times, expected 3"
            )

    def test_sampling_with_multi_turn(self, cli: AIPerfCLI):
        """Test sampling strategies with multi-turn conversations."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --dataset-sampling-strategy shuffle \
                --random-seed 42 \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Should have 10 sessions × 3 turns = 30 requests
        assert len(result.jsonl) == 30

        # Verify turn indices are sequential within each session
        analyzer = CreditFlowAnalyzer(result.runner_result)
        assert analyzer.turn_indices_sequential()

        # Verify each session has exactly 3 turns
        assert analyzer.session_credits_match(expected_turns=3)

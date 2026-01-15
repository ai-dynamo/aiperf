# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup phase tests with interaction coverage.

Warmup phase is critical for:
- KV cache warm-up
- Connection pool initialization
- Latency stabilization
- Avoiding cold-start bias in benchmark metrics

These tests verify:
1. Basic warmup functionality (request-count, duration)
2. Warmup to profiling phase transition (ordering, credit isolation)
3. Warmup interactions with multi-turn conversations and cancellation

CRITICAL: Warmup and profiling phases have SEPARATE credit tracking.
Each phase should balance independently.
"""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from tests.component_integration.timing.conftest import defaults
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestWarmupBasic:
    """Basic warmup phase functionality tests."""

    def test_warmup_request_count_completes(self, cli: AIPerfCLI):
        """Test basic warmup with request count.

        Scenario:
        - Warmup: 20 requests
        - Profiling: 30 requests
        - Verify both phases complete
        - Verify separate credit tracking
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 30 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Total requests = warmup (20) + profiling (30) = 50
        # But result.request_count only counts profiling phase
        assert result.request_count == 30

        runner = result.runner_result

        # Verify we have credits from BOTH phases
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_credits) == 20, (
            f"Expected 20 warmup credits, got {len(warmup_credits)}"
        )
        assert len(profiling_credits) == 30, (
            f"Expected 30 profiling credits, got {len(profiling_credits)}"
        )

        # Verify both phases balanced independently
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]

        warmup_returns = [
            p for p in return_payloads if p.payload.credit.phase == CreditPhase.WARMUP
        ]
        profiling_returns = [
            p
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.PROFILING
        ]

        assert len(warmup_returns) == 20
        assert len(profiling_returns) == 30

    def test_warmup_duration_completes(self, cli: AIPerfCLI):
        """Test warmup with duration instead of request count.

        Scenario:
        - Warmup: 0.3 seconds at 200 QPS → ~60 requests
        - Profiling: 25 requests
        - Verify duration stops warmup, profiling continues
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 25 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-duration 0.3
        """

        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == 25  # Profiling only

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup duration 0.3s at 200 QPS → ~60 requests (widened for CI jitter)
        assert 40 <= len(warmup_credits) <= 80, (
            f"Expected ~60 warmup credits, got {len(warmup_credits)}"
        )
        assert len(profiling_credits) == 25


@pytest.mark.component_integration
class TestWarmupPhaseTransition:
    """Tests for warmup → profiling phase transition.

    The transition should be seamless with no credit loss or duplication.
    """

    def test_warmup_to_profiling_transition_seamless(self, cli: AIPerfCLI):
        """Test seamless transition from warmup to profiling.

        Scenario:
        - Warmup phase completes
        - Profiling phase starts immediately
        - No credit loss, no overlap
        - Credits balanced per phase
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 20 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 15
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result

        # Get all credits sorted by capture timestamp
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]
        credit_payloads.sort(key=lambda p: p.timestamp_ns)

        # Verify phase ordering (warmup first, then profiling)
        phases = [p.payload.phase for p in credit_payloads]
        warmup_end_idx = phases.index(CreditPhase.PROFILING)

        assert all(p == CreditPhase.WARMUP for p in phases[:warmup_end_idx])
        assert all(p == CreditPhase.PROFILING for p in phases[warmup_end_idx:])

        # Verify counts
        assert warmup_end_idx == 15  # 15 warmup credits
        assert len(phases) - warmup_end_idx == 20  # 20 profiling credits

    def test_credits_isolated_per_phase(self, cli: AIPerfCLI):
        """Test that warmup and profiling credits are isolated.

        Scenario:
        - Warmup and profiling both run
        - Credit IDs should restart at 0 for each phase
        - No cross-phase contamination
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 12 \
                --request-rate 250 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 10
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p.payload for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p.payload
            for p in credit_payloads
            if p.payload.phase == CreditPhase.PROFILING
        ]

        # Credit IDs should restart for each phase
        warmup_ids = {c.id for c in warmup_credits}
        profiling_ids = {c.id for c in profiling_credits}

        # Both should start from 0
        assert min(warmup_ids) == 0
        assert min(profiling_ids) == 0

        # Both should be sequential
        assert warmup_ids == set(range(10))
        assert profiling_ids == set(range(12))


@pytest.mark.component_integration
class TestWarmupInteractions:
    """Tests for warmup interactions with other features.

    These tests focus on how warmup interacts with:
    - Multi-turn conversations (session isolation)
    - Cancellation (disabled during warmup)
    """

    def test_warmup_with_multi_turn_conversations(self, cli: AIPerfCLI):
        """Test warmup + multi-turn interaction.

        Scenario:
        - Warmup: 10 sessions × 3 turns = 30 credits
        - Profiling: 15 sessions × 3 turns = 45 credits
        - Verify turn indices sequential per phase
        - Verify session isolation between phases
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 15 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --request-rate 250 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 10
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Profiling: 15 sessions × 3 turns = 45 requests
        assert result.request_count == 45

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p.payload for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p.payload
            for p in credit_payloads
            if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup: 10 sessions × 3 turns = 30
        assert len(warmup_credits) == 30
        assert len(profiling_credits) == 45

        # Verify warmup sessions are different from profiling sessions
        warmup_session_ids = {c.x_correlation_id for c in warmup_credits}
        profiling_session_ids = {c.x_correlation_id for c in profiling_credits}

        # Session IDs should be different between phases
        assert warmup_session_ids.isdisjoint(profiling_session_ids), (
            "Warmup and profiling should use different sessions"
        )

    def test_warmup_cancellation_disabled(self, cli: AIPerfCLI):
        """Test that cancellation is disabled during warmup phase.

        Scenario:
        - Cancellation rate 50% configured
        - Warmup: Should have 0 cancellations
        - Profiling: Should have ~50% cancellations
        - Verify warmup immune to cancellation
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-count 20 \
                --request-rate 300 \
                --request-rate-mode constant \
                --osl 5 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --warmup-request-count 20 \
                --request-cancellation-rate 50.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]

        warmup_returns = [
            p.payload
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.WARMUP
        ]
        profiling_returns = [
            p.payload
            for p in return_payloads
            if p.payload.credit.phase == CreditPhase.PROFILING
        ]

        # Warmup: No errors (cancellation disabled)
        warmup_errors = sum(1 for r in warmup_returns if r.error is not None)
        assert warmup_errors == 0, f"Warmup should have 0 errors, got {warmup_errors}"

        # Profiling: Should have errors (~50% with seed 42)
        profiling_errors = sum(1 for r in profiling_returns if r.error is not None)
        assert profiling_errors > 0, (
            "Profiling should have some errors with 50% cancellation rate"
        )

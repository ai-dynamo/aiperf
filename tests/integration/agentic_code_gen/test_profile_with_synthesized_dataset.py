# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: synthesize an agentic_code_gen dataset and run it through aiperf profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestAgenticCodeGenProfile:
    """End-to-end: synthesize -> profile with mock server."""

    async def test_synthesized_dataset_runs_through_profile(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Synthesize a small dataset and run it through aiperf profile with session concurrency."""
        from aiperf.dataset.agentic_code_gen.models import SessionDistributionConfig
        from aiperf.dataset.agentic_code_gen.session_synthesizer import (
            SessionSynthesizer,
        )
        from aiperf.dataset.agentic_code_gen.writer import write_dataset

        config = SessionDistributionConfig(max_prompt_tokens=10_000)
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(5)
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, config, seed=42, config_name="default")

        jsonl_path = run_dir / "dataset.jsonl"
        total_turns = sum(len(s.turns) for s in sessions)

        session_concurrency = len(sessions)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --tokenizer {defaults.tokenizer} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {jsonl_path} \
                --custom-dataset-type mooncake_trace \
                --request-count {total_turns} \
                --concurrency {session_concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == total_turns
        assert result.has_all_outputs_except_inputs
        assert result.inputs is None, "trace datasets (mooncake_trace) skip inputs.json"

        # inputs.json is skipped for trace datasets, so verify all 5 sessions
        # loaded with the correct multi-turn structure from the per-record
        # metadata (one record per turn) instead.
        assert result.jsonl is not None
        turns_per_session: dict[str, int] = {}
        for record in result.jsonl:
            conversation_id = record.metadata.conversation_id
            assert conversation_id is not None
            turns_per_session[conversation_id] = (
                turns_per_session.get(conversation_id, 0) + 1
            )
        assert len(turns_per_session) == len(sessions)
        assert sorted(turns_per_session.values()) == sorted(
            len(synth_session.turns) for synth_session in sessions
        )

    async def test_synthesized_dataset_with_shared_partial_prefix_blocks(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """Regression: a synthesized dataset whose sessions have small L2 tails
        (landing on the shared-prefix block boundary) must still replay through
        the real trace loader.

        Before the fix the synthesizer absorbed a small L2 remainder into the
        last shared L1.5 block, so that shared hash_id was a partial final block
        in one session and a full block in another. Trace reconstruction then
        raised ``ConfigurationError`` at Configure Profiling time and the whole
        run aborted. This config (small block size, tiny L2 relative to a
        block, few groups) reliably produces that shared-partial-block case.
        """
        from aiperf.dataset.agentic_code_gen.distributions import (
            lognormal_from_mean_median,
        )
        from aiperf.dataset.agentic_code_gen.models import (
            CacheLayerConfig,
            Layer15GroupConfig,
            SessionDistributionConfig,
        )
        from aiperf.dataset.agentic_code_gen.session_synthesizer import (
            SessionSynthesizer,
        )
        from aiperf.dataset.agentic_code_gen.writer import write_dataset

        config = SessionDistributionConfig(
            block_size=64,
            max_prompt_tokens=6_000,
            reset=None,
            cache=CacheLayerConfig(
                layer1_tokens=1_000,
                layer1_5_tokens=500,
                layer2=lognormal_from_mean_median(mean=40, median=30),
                layer1_5_groups=Layer15GroupConfig(num_groups=3, zipf_alpha=1.2),
            ),
        )
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(8)
        run_dir = tmp_path / "run"
        write_dataset(sessions, run_dir, config, seed=42, config_name="default")

        jsonl_path = run_dir / "dataset.jsonl"
        total_turns = sum(len(s.turns) for s in sessions)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --tokenizer {defaults.tokenizer} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {jsonl_path} \
                --custom-dataset-type mooncake_trace \
                --isl-block-size {config.block_size} \
                --request-count {total_turns} \
                --concurrency {len(sessions)} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == total_turns
        assert result.has_all_outputs_except_inputs

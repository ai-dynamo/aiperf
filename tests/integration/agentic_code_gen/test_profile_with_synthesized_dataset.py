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
        assert result.has_all_outputs

        # Verify all 5 sessions were loaded with correct multi-turn structure
        assert result.inputs is not None
        assert len(result.inputs.data) == len(sessions)
        for input_session, synth_session in zip(
            result.inputs.data, sessions, strict=False
        ):
            assert len(input_session.payloads) == len(synth_session.turns)

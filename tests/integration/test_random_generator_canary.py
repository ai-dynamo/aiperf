# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary test for random generator consistency.

This test serves as a regression detector for the decoupled random generators.
It compares the exact output of a seeded run against a known reference, ensuring
that any changes to the codebase don't silently break determinism.
"""

import json
from pathlib import Path

import pytest

from aiperf.common.utils import load_json_str
from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestRandomGeneratorCanary:
    """Canary tests for random generator regression detection."""

    CANARY_SEED = 42
    REFERENCE_FILE = Path(__file__).parent / "assets" / "canary_reference_inputs.json"

    async def test_random_generator_canary(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify random generator produces exact output matching reference.

        This test runs aiperf with a fixed seed and compares the generated inputs.json
        against a reference file. Any change in the random generation that affects
        output will cause this test to fail, alerting us to potential regressions.

        Note: session_id (UUIDs) are excluded from comparison as they are not seeded.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model-names "openai/gpt-oss-20b,openai/gpt-oss-120b" \
                --model-selection-strategy random \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 20 \
                --concurrency 2 \
                --random-seed {self.CANARY_SEED} \
                --prompt-input-tokens-mean 100 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 20 \
                --prompt-output-tokens-mean 50 \
                --prompt-output-tokens-stddev 5 \
                --audio-length-mean 0.05 \
                --audio-length-stddev 0.01 \
                --workers-max 2 \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == 20
        assert result.inputs is not None

        # Load reference data
        if not self.REFERENCE_FILE.exists():
            # Generate reference file if it doesn't exist (for first-time setup)
            self._save_reference(result.inputs.model_dump())
            pytest.fail(
                f"Reference file created at {self.REFERENCE_FILE}. "
                "Run test again to validate against reference."
            )

        reference_data = load_json_str(self.REFERENCE_FILE.read_text())
        current_data = result.inputs.model_dump()
        self._assert_inputs_match(reference_data, current_data)

    def _assert_inputs_match(self, reference: dict, current: dict) -> None:
        """Compare inputs while excluding session_id (UUIDs).

        Args:
            reference: Reference inputs data
            current: Current run inputs data

        Raises:
            AssertionError: If payloads don't match exactly
        """
        assert "data" in reference
        assert "data" in current

        ref_sessions = reference["data"]
        cur_sessions = current["data"]

        assert len(ref_sessions) == len(cur_sessions), (
            f"Session count mismatch: expected {len(ref_sessions)}, "
            f"got {len(cur_sessions)}"
        )

        for i, (ref_session, cur_session) in enumerate(
            zip(ref_sessions, cur_sessions, strict=True)
        ):
            # Compare payloads exactly (this is the critical part)
            assert ref_session["payloads"] == cur_session["payloads"], (
                f"Session {i}: Payloads don't match!\n"
                f"This indicates a regression in random generation.\n"
                f"Reference: {ref_session['payloads']}\n"
                f"Current:   {cur_session['payloads']}"
            )

    def _save_reference(self, data: dict) -> None:
        """Save reference inputs.json for future comparisons.

        Args:
            data: Inputs data to save as reference
        """
        self.REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.REFERENCE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

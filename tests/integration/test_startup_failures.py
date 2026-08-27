# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for startup failure scenarios.

These tests verify that aiperf exits with a non-zero exit code (rather than hanging)
when services fail during startup or configuration.
"""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestStartupFailures:
    """Tests for startup failure scenarios that should exit with errors."""

    async def test_invalid_model_name_exits_with_error(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that an invalid model name causes aiperf to exit with an error.

        This test verifies the fail-fast behavior: when a service (like DatasetManager)
        fails during configuration, the system should exit promptly with a non-zero
        exit code rather than hanging indefinitely.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model this-model-does-not-exist-and-will-fail \
                --tokenizer this-model-does-not-exist-and-will-fail \
                --url {aiperf_mock_server.url} \
                --request-count 10 \
                --concurrency 2
            """,
            assert_success=False,
            timeout=60.0,  # Should fail fast, not wait for full timeout
        )
        assert result.exit_code != 0, (
            "Expected non-zero exit code when model/tokenizer fails to load"
        )

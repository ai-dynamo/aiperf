# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the `--wait-for-model` readiness probe.

Covers:
- success immediately (models endpoint ready from t=0)
- success after N retries (models endpoint returns empty data until delay elapses)
- timeout failure (requested model never appears)
- 404 fallback (models endpoint disabled; probe accepts 2xx on base URL)
"""

import pytest

from tests.harness.utils import AIPerfCLI


@pytest.mark.integration
@pytest.mark.asyncio
class TestWaitForModel:
    """Tests for `aiperf profile --wait-for-model`."""

    async def test_wait_for_model_success_immediate(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """With no configured delay, /v1/models lists the model from the start
        and the probe returns on the first attempt."""
        async with mock_server_factory(
            fast=True, workers=1, default_model="mock-model"
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --concurrency 1
                    --request-count 1
                    --workers-max 1
                    --ui simple
                    --wait-for-model
                    --wait-for-model-timeout 30
                    --wait-for-model-interval 1
                """,
                timeout=120.0,
            )
            assert result.exit_code == 0
            combined = f"{result.stdout}\n{result.stderr}\n{result.log}"
            assert "Model 'mock-model' ready" in combined

    async def test_wait_for_model_success_after_retries(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """With models_ready_delay_seconds>0, the probe sees an empty
        data list on early attempts and must retry until the model appears."""
        async with mock_server_factory(
            fast=True,
            workers=1,
            default_model="mock-model",
            models_ready_delay_seconds=2.0,
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --concurrency 1
                    --request-count 1
                    --workers-max 1
                    --ui simple
                    --wait-for-model
                    --wait-for-model-timeout 20
                    --wait-for-model-interval 0.5
                """,
                timeout=120.0,
            )
            assert result.exit_code == 0
            # At least one retry log line should have fired before the model appeared.
            combined = f"{result.stdout}\n{result.stderr}\n{result.log}"
            assert "not yet in" in combined
            assert "Model 'mock-model' ready" in combined

    async def test_wait_for_model_timeout(self, cli: AIPerfCLI, mock_server_factory):
        """If the requested model id never appears in /v1/models, the probe
        must exit non-zero and the error must reference the model and URL."""
        async with mock_server_factory(
            fast=True, workers=1, default_model="mock-model"
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile
                    --model this-model-is-never-served
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --concurrency 1
                    --request-count 1
                    --workers-max 1
                    --ui simple
                    --wait-for-model
                    --wait-for-model-timeout 3
                    --wait-for-model-interval 0.5
                """,
                timeout=60.0,
                assert_success=False,
            )
            assert result.exit_code != 0
            combined = f"{result.stdout}\n{result.stderr}\n{result.log}"
            assert "this-model-is-never-served" in combined
            assert server.url in combined
            assert "Timed out" in combined

    async def test_wait_for_model_404_fallback(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """When /v1/models returns 404, the probe must fall back to a base-URL
        GET and accept a 2xx as 'server is up'."""
        async with mock_server_factory(
            fast=True,
            workers=1,
            default_model="mock-model",
            disable_models_endpoint=True,
        ) as server:
            result = await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --concurrency 1
                    --request-count 1
                    --workers-max 1
                    --ui simple
                    --wait-for-model
                    --wait-for-model-timeout 15
                    --wait-for-model-interval 1
                """,
                timeout=120.0,
            )
            assert result.exit_code == 0
            combined = f"{result.stdout}\n{result.stderr}\n{result.log}"
            assert "accepting as ready" in combined

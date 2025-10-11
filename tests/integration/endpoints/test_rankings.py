# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestRankingsEndpoint:
    """Tests for /v1/ranking endpoint."""

    async def test_basic_rankings(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, create_rankings_dataset
    ):
        """Basic rankings with custom dataset."""
        dataset_path = create_rankings_dataset(5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10

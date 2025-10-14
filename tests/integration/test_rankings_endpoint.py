# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/ranking endpoint."""

from pathlib import Path

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer
from tests.integration.utils import create_rankings_dataset


@pytest.mark.integration
@pytest.mark.asyncio
class TestRankingsEndpoint:
    """Tests for /v1/ranking endpoint."""

    async def test_basic_rankings(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer, tmp_path: Path
    ):
        """Basic rankings with custom dataset."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/nv-rerank-qa-mistral-4b \
                --url {fakeai_server.url} \
                --tokenizer gpt2 \
                --endpoint-type rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 10

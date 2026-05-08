# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for embedding endpoints."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestEmbeddingsEndpoint:
    """Tests for embedding endpoints."""

    @pytest.mark.parametrize("endpoint_type", ["embeddings", "cohere_embeddings"])
    def test_basic_embeddings(self, cli: AIPerfCLI, endpoint_type: str):
        """Basic embeddings request."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nomic-ai/nomic-embed-text-v1.5 \
                --tokenizer gpt2 \
                --endpoint-type {endpoint_type} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        # Embeddings don't stream, so streaming metrics should be absent
        assert result.json.time_to_first_token is None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ranking endpoints."""

from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI
from tests.integration.utils import create_rankings_dataset


@pytest.mark.component_integration
class TestRankingsEndpoint:
    """Tests for ranking-type endpoints."""

    def test_nim_rankings(self, cli: AIPerfCLI, tmp_path: Path):
        """Basic rankings test for NIM Rankings endpoint (/v1/ranking)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/nv-rerank-qa-mistral-4b \
                --tokenizer gpt2 \
                --endpoint-type nim_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    def test_hf_tei_rankings(self, cli: AIPerfCLI, tmp_path: Path):
        """Test for HuggingFace TEI Rankings endpoint (/rerank)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model BAAI/bge-reranker-base  \
                --tokenizer bert-base-uncased \
                --endpoint-type hf_tei_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    def test_cohere_rankings(self, cli: AIPerfCLI, tmp_path: Path):
        """Test for Cohere Rankings endpoint (/v2/rerank)."""
        dataset_path = create_rankings_dataset(tmp_path, 5)

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model BAAI/bge-reranker-base \
                --endpoint-type cohere_rankings \
                --input-file {dataset_path} \
                --custom-dataset-type single_turn \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

    def test_synthetic_nim_rankings(self, cli: AIPerfCLI):
        """Synthetic dataset test for NIM Rankings endpoint (/v1/ranking)."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/nv-rerank-qa-mistral-4b \
                --tokenizer gpt2 \
                --endpoint-type nim_rankings \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --rankings-passages-mean 6 \
                --rankings-passages-stddev 2 \
                --rankings-passages-prompt-token-mean 32 \
                --rankings-passages-prompt-token-stddev 8 \
                --rankings-query-prompt-token-mean 16 \
                --rankings-query-prompt-token-stddev 4 \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count

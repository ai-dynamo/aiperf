# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI validation tests for non-tokenizing endpoint constraints (image_retrieval)."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

BASE_CMD = (
    "aiperf profile"
    " --model nvidia/page-elements-v2"
    " --endpoint-type image_retrieval"
    " --endpoint /v1/image/infer"
    " --image-width-mean 64"
    " --image-height-mean 64"
    f" --request-count {defaults.request_count}"
    f" --concurrency {defaults.concurrency}"
    f" --workers-max {defaults.workers_max}"
    f" --ui {defaults.ui}"
)


@pytest.mark.component_integration
class TestNonTokenizingEndpointTextValidation:
    """CLI rejects text-related options for non-tokenizing endpoints."""

    def test_rejects_synthetic_input_tokens_mean(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --synthetic-input-tokens-mean 128",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "--synthetic-input-tokens-mean" in result.stderr

    def test_rejects_synthetic_input_tokens_stddev(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --synthetic-input-tokens-stddev 32",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "--synthetic-input-tokens-stddev" in result.stderr

    def test_rejects_batch_size_text(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --batch-size-text 4",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "--batch-size-text" in result.stderr

    def test_rejects_sequence_distribution(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --sequence-distribution 128,64:50;256,128:50",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "--sequence-distribution" in result.stderr

    def test_rejects_prefix_prompt_options(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --prefix-prompt-pool-size 5 --prefix-prompt-length 100",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "Prefix prompt options" in result.stderr


@pytest.mark.component_integration
class TestNonTokenEndpointTokenizerValidation:
    """CLI rejects tokenizer options for non-token endpoints."""

    def test_rejects_tokenizer(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --tokenizer some-model",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "Tokenizer options" in result.stderr

    def test_rejects_tokenizer_trust_remote_code(self, cli: AIPerfCLI):
        result = cli.run_sync(
            f"{BASE_CMD} --tokenizer-trust-remote-code",
            assert_success=False,
        )
        assert result.exit_code == 1
        assert "Tokenizer options" in result.stderr

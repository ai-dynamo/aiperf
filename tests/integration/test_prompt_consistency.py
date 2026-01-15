# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for random prompt generation consistency.

This test ensures that randomly generated prompt texts remain consistent across
different configuration changes when using the same seed. The goal is to verify
that the random text generation is decoupled from other configuration parameters.
"""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


def extract_prompt_texts(result) -> list[str]:
    """Extract all prompt text content from payloads.

    Args:
        result: AIPerfResults object containing inputs data

    Returns:
        List of all text content from prompts in order
    """
    texts = []
    for session in result.inputs.data:
        for payload in session.payloads:
            if "messages" in payload:
                # Chat format
                for message in payload["messages"]:
                    if isinstance(message.get("content"), str):
                        texts.append(message["content"])
                    elif isinstance(message.get("content"), list):
                        # Multimodal content
                        for item in message["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                texts.append(item["text"])
            elif "prompt" in payload:
                # Completions format
                texts.append(payload["prompt"])
    return texts


@pytest.mark.integration
@pytest.mark.asyncio
class TestPromptConsistency:
    """Tests for random prompt text consistency across configuration changes."""

    CONSISTENCY_SEED = 12345

    async def test_prompt_consistency_with_multimodal_additions(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify prompt texts are identical when adding audio/images.

        Adding multimodal content (audio/images) should not affect the randomly
        generated text portions of prompts.
        """
        # Run without multimodal content
        result_text_only = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 90 \
                --prompt-input-tokens-stddev 8 \
                --num-dataset-entries 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with audio and images
        result_multimodal = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 90 \
                --prompt-input-tokens-stddev 8 \
                --num-dataset-entries 10 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --audio-length-mean 0.1 \
                --audio-length-stddev 0.02 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_text_only = extract_prompt_texts(result_text_only)
        texts_multimodal = extract_prompt_texts(result_multimodal)

        assert len(texts_text_only) == len(texts_multimodal), (
            "Prompt count should be identical"
        )
        assert texts_text_only == texts_multimodal, (
            "Prompt texts should be identical even when audio/images are added"
        )

    async def test_prompt_consistency_comprehensive_same_endpoint(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Comprehensive test combining multiple configuration changes.

        This test changes multiple parameters at once (keeping the same
        endpoint type) to ensure the prompt text generation is truly
        decoupled from configuration.

        Note: Both runs explicitly set the same tokenizer because different
        tokenizers tokenize the corpus differently, resulting in different
        prompts. The goal here is to test that OTHER configuration changes
        (concurrency, multimodal, etc.) don't affect prompt generation.
        """
        # Baseline run with minimal config
        result_baseline = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --tokenizer {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 1 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 100 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with many parameters changed (but same endpoint type and tokenizer)
        result_modified = await cli.run(
            f"""
            aiperf profile \
                --model-names "openai/gpt-oss-20b,openai/gpt-oss-120b,Qwen/Qwen3-0.6B" \
                --model-selection-strategy random \
                --tokenizer {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 3 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 100 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --prompt-output-tokens-mean 150 \
                --prompt-output-tokens-stddev 15 \
                --image-width-mean 256 \
                --image-height-mean 256 \
                --audio-length-mean 0.2 \
                --audio-length-stddev 0.05 \
                --workers-max 2 \
                --ui {defaults.ui}
            """
        )

        texts_baseline = extract_prompt_texts(result_baseline)
        texts_modified = extract_prompt_texts(result_modified)

        assert len(texts_baseline) == len(texts_modified), (
            "Prompt count should be identical"
        )
        assert texts_baseline == texts_modified, (
            "Prompt texts should be identical even with comprehensive config changes"
        )

    async def test_prompt_consistency_with_dataset_sampling_strategies(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify dataset sampling strategy doesn't affect prompt generation.

        The dataset sampling strategy (sequential, random, shuffle) determines
        how to sample from the dataset, but should not affect the underlying
        prompt text generation itself.
        """
        # Run with sequential sampling
        result_sequential = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 80 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --dataset-sampling-strategy sequential \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with shuffle sampling
        result_shuffle = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 80 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --dataset-sampling-strategy shuffle \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_sequential = extract_prompt_texts(result_sequential)
        texts_shuffle = extract_prompt_texts(result_shuffle)

        # Note: The ORDER might be different due to shuffling, but the
        # set of generated texts should be the same
        assert len(texts_sequential) == len(texts_shuffle), (
            "Prompt count should be identical"
        )
        assert set(texts_sequential) == set(texts_shuffle), (
            "Dataset sampling strategy should not affect which prompts are generated, "
            "only the order they are used"
        )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for random prompt generation consistency.

This test ensures that randomly generated prompt texts remain consistent across
different configuration changes when using the same seed. The goal is to verify
that the random text generation is decoupled from other configuration parameters.
"""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


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

    async def test_prompt_consistency_across_model_selection_strategies(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify prompt texts are identical across different model selection strategies.

        Even when changing from round-robin to random model selection, the
        randomly generated prompt texts should remain the same with the same seed.
        """
        # Run with round-robin model selection
        result_round_robin = await cli.run(
            f"""
            aiperf profile \
                --model-names "openai/gpt-oss-20b,openai/gpt-oss-120b" \
                --model-selection-strategy round_robin \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 15 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 80 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 15 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with random model selection
        result_random = await cli.run(
            f"""
            aiperf profile \
                --model-names "openai/gpt-oss-20b,openai/gpt-oss-120b" \
                --model-selection-strategy random \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 15 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 80 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 15 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_round_robin = extract_prompt_texts(result_round_robin)
        texts_random = extract_prompt_texts(result_random)

        assert len(texts_round_robin) == len(texts_random), (
            "Prompt count should be identical"
        )
        assert texts_round_robin == texts_random, (
            "Prompt texts should be identical regardless of model selection strategy"
        )

    async def test_prompt_consistency_across_max_tokens_changes(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify prompt texts are identical when max_tokens parameters change.

        Changing output token configuration should not affect the randomly
        generated input prompt texts.
        """
        # Run with one max_tokens configuration
        result_tokens_50 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 12 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 100 \
                --prompt-input-tokens-stddev 5 \
                --prompt-output-tokens-mean 50 \
                --prompt-output-tokens-stddev 5 \
                --num-dataset-entries 12 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with different max_tokens configuration
        result_tokens_200 = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 12 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 100 \
                --prompt-input-tokens-stddev 5 \
                --prompt-output-tokens-mean 200 \
                --prompt-output-tokens-stddev 20 \
                --num-dataset-entries 12 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_50 = extract_prompt_texts(result_tokens_50)
        texts_200 = extract_prompt_texts(result_tokens_200)

        assert len(texts_50) == len(texts_200), "Prompt count should be identical"
        assert texts_50 == texts_200, (
            "Prompt texts should be identical regardless of max_tokens configuration"
        )

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

    async def test_prompt_consistency_with_concurrency_changes(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify prompt texts are identical when concurrency settings change.

        Changing concurrency should not affect the randomly generated
        prompt texts.
        """
        # Run with low concurrency
        result_low_concurrency = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 8 \
                --concurrency 1 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 75 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 8 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with higher concurrency
        result_high_concurrency = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 8 \
                --concurrency 4 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 75 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 8 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_low = extract_prompt_texts(result_low_concurrency)
        texts_high = extract_prompt_texts(result_high_concurrency)

        assert len(texts_low) == len(texts_high), "Prompt count should be identical"
        assert texts_low == texts_high, (
            "Prompt texts should be identical regardless of concurrency"
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

    async def test_prompt_consistency_with_different_dataset_sizes(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify first N prompts are identical when generating different dataset sizes.

        When generating 10 entries vs 20 entries with the same seed, the first
        10 entries should be identical in both runs. This ensures the random
        generation is sequential and deterministic.
        """
        # Generate 10 entries
        result_10_entries = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 90 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Generate 20 entries with same seed
        result_20_entries = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 20 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 90 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 20 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_10 = extract_prompt_texts(result_10_entries)
        texts_20 = extract_prompt_texts(result_20_entries)

        # Verify we got the expected counts
        assert len(texts_10) == 10, f"Expected 10 prompts, got {len(texts_10)}"
        assert len(texts_20) == 20, f"Expected 20 prompts, got {len(texts_20)}"

        # Verify the first 10 prompts are identical
        texts_20_first_10 = texts_20[:10]
        assert texts_10 == texts_20_first_10, (
            "First 10 prompts should be identical when generating 10 vs 20 entries. "
            "This ensures deterministic sequential generation."
        )

    async def test_prompt_consistency_with_warmup_requests(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify warmup requests don't affect prompt text generation.

        Warmup requests should not consume from the random generator state
        used for prompt text generation.
        """
        # Run without warmup
        result_no_warmup = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 85 \
                --prompt-input-tokens-stddev 8 \
                --num-dataset-entries 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with warmup
        result_with_warmup = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 85 \
                --prompt-input-tokens-stddev 8 \
                --num-dataset-entries 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_no_warmup = extract_prompt_texts(result_no_warmup)
        texts_with_warmup = extract_prompt_texts(result_with_warmup)

        assert len(texts_no_warmup) == len(texts_with_warmup), (
            "Prompt count should be identical"
        )
        assert texts_no_warmup == texts_with_warmup, (
            "Warmup requests should not affect prompt text generation"
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

    async def test_prompt_consistency_with_request_count_mismatch(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify prompt consistency when request count exceeds dataset entries.

        When request-count > num-dataset-entries, the system cycles through
        the dataset. The generated prompts should still be consistent.
        """
        # Generate 10 dataset entries
        result_10_entries = await cli.run(
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

        # Generate 5 dataset entries (5 unique prompts will be created)
        result_5_entries = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 5 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 90 \
                --prompt-input-tokens-stddev 8 \
                --num-dataset-entries 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_10 = extract_prompt_texts(result_10_entries)
        texts_5 = extract_prompt_texts(result_5_entries)

        # Verify counts
        assert len(texts_10) == 10
        assert len(texts_5) == 5

        # The first 5 prompts from the 10-entry run should match the 5-entry run
        texts_10_first_5 = texts_10[:5]
        assert texts_10_first_5 == texts_5, (
            "First 5 prompts should be identical regardless of num-dataset-entries"
        )

    async def test_prompt_consistency_with_random_image_format(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Verify random image format selection doesn't affect prompt texts.

        When using --image-format random, the format selection uses a separate
        random generator and should not affect prompt text generation.
        """
        # Run with fixed image format
        result_fixed_format = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 85 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format png \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        # Run with random image format
        result_random_format = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --random-seed {self.CONSISTENCY_SEED} \
                --prompt-input-tokens-mean 85 \
                --prompt-input-tokens-stddev 10 \
                --num-dataset-entries 10 \
                --image-width-mean 128 \
                --image-height-mean 128 \
                --image-format random \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        texts_fixed = extract_prompt_texts(result_fixed_format)
        texts_random = extract_prompt_texts(result_random_format)

        assert len(texts_fixed) == len(texts_random), "Prompt count should be identical"
        assert texts_fixed == texts_random, (
            "Image format randomization should not affect prompt text generation"
        )

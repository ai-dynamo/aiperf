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

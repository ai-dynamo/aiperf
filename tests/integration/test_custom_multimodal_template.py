# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for custom multi-modal endpoint using template format."""

from pathlib import Path

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestCustomMultimodalTemplate:
    """Tests for custom multi-modal endpoint using custom template."""

    async def test_custom_multimodal_simple(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Test custom multi-modal endpoint with simple text only first."""
        # Start with a super simple template to debug
        template = """{
    "modality_bundle": {
        "text_fragments": {{ texts|tojson }}
    },
    "inference_params": {
        "model_id": {{ model|tojson }}
    }
}"""

        # Write template to a file to avoid shell escaping issues
        template_file = tmp_path / "custom_template_simple.json"
        template_file.write_text(template)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url}/v1/custom-multimodal \
                --endpoint-type template \
                --extra-inputs payload_template:{template_file} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    async def test_custom_multimodal_with_images_and_audio(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Test custom multi-modal endpoint with images and audio using custom template."""
        # Custom jinja2 template that matches the custom endpoint format
        template = """{
    "modality_bundle": {
        "text_fragments": {{ texts|tojson }},
        "visual_assets": {
            "images": {{ images|tojson }}
        },
        "audio_streams": {{ audios|tojson }}
    },
    "inference_params": {
        "model_id": {{ model|tojson }},
        "sampling_config": {
            "max_tokens": {{ max_tokens|tojson }}
        }
    }
}"""

        # Write template to a file to avoid shell escaping issues
        template_file = tmp_path / "custom_template.json"
        template_file.write_text(template)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url}/v1/custom-multimodal \
                --endpoint-type template \
                --extra-inputs payload_template:{template_file} \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --synthetic-input-tokens-mean 50 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_all_outputs

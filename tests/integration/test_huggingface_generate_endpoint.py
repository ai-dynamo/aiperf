# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestHuggingFaceGenerateEndpoint:
    """Integration tests for huggingface_generate endpoint."""

    async def _create_input_file(self, lines: list[str], tmp_path: Path) -> Path:
        """Helper to create a temporary input file with given text lines."""
        input_file = tmp_path / "inputs.jsonl"
        input_file.write_text("".join(f'{{"text": "{line}"}}\n' for line in lines))
        return input_file

    async def _run_profile(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        streaming: bool,
        input_file: Path | None = None,
    ):
        """Helper to run CLI profile for huggingface_generate."""
        stream_flag = "--streaming" if streaming else ""
        dataset_flag = ""
        input_flag = ""

        if input_file:
            dataset_flag = "--custom-dataset-type single_turn"
            input_flag = f"--input-file {input_file}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --url {aiperf_mock_server.url} \
                --endpoint-type huggingface_generate \
                {stream_flag} \
                {input_flag} \
                {dataset_flag} \
                --request-count {defaults.request_count}
            """
        )
        assert result.request_count == defaults.request_count
        return result

    async def test_synthetic_non_streaming(self, cli, aiperf_mock_server):
        """Synthetic (no input file) non-streaming run."""
        result = await self._run_profile(cli, aiperf_mock_server, streaming=False)
        assert not result.has_streaming_metrics

    async def test_synthetic_streaming(self, cli, aiperf_mock_server):
        """Synthetic (no input file) streaming run."""
        result = await self._run_profile(cli, aiperf_mock_server, streaming=True)
        assert result.has_streaming_metrics

    async def test_file_input_non_streaming(self, cli, aiperf_mock_server, tmp_path):
        """File input non-streaming run."""
        input_file = await self._create_input_file(
            ["Hello TinyLlama!", "Tell me a joke."], tmp_path
        )
        result = await self._run_profile(
            cli, aiperf_mock_server, streaming=False, input_file=input_file
        )
        assert not result.has_streaming_metrics

    async def test_file_input_streaming(self, cli, aiperf_mock_server, tmp_path):
        """File input streaming run."""
        input_file = await self._create_input_file(
            ["Stream something poetic.", "Give me a haiku."], tmp_path
        )
        result = await self._run_profile(
            cli, aiperf_mock_server, streaming=True, input_file=input_file
        )
        assert result.has_streaming_metrics

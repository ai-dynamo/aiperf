# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for huggingface_generate endpoint."""

from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestHuggingFaceGenerateEndpoint:
    """Tests for huggingface_generate endpoint."""

    def _create_input_file(self, tmp_path: Path) -> Path:
        """Helper to create a temporary input file."""
        input_file = tmp_path / "inputs.jsonl"
        input_file.write_text(
            '{"text": "Hello TinyLlama!"}\n{"text": "Tell me a joke."}\n'
        )
        return input_file

    def _run_profile(
        self,
        cli: AIPerfCLI,
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

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --endpoint-type huggingface_generate \
                {stream_flag} \
                {input_flag} \
                {dataset_flag} \
                --request-count {defaults.request_count} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        return result

    @pytest.mark.parametrize(
        ("streaming", "use_file_input"),
        [
            (False, False),  # synthetic non-streaming
            (True, False),  # synthetic streaming
            (False, True),  # file input non-streaming
            (True, True),  # file input streaming
        ],  # fmt: skip
        ids=[
            "synthetic_non_streaming",
            "synthetic_streaming",
            "file_input_non_streaming",
            "file_input_streaming",
        ],
    )
    def test_huggingface_generate(
        self,
        cli: AIPerfCLI,
        tmp_path: Path,
        streaming: bool,
        use_file_input: bool,
    ):
        """Test huggingface_generate endpoint with various configurations."""
        input_file = self._create_input_file(tmp_path) if use_file_input else None
        result = self._run_profile(cli, streaming=streaming, input_file=input_file)

        if streaming:
            assert result.has_streaming_metrics
        else:
            assert not result.has_streaming_metrics

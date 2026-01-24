# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Mooncake trace custom dataset type."""

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import create_mooncake_trace_file


@pytest.mark.integration
@pytest.mark.asyncio
class TestMooncakeTraceIntegration:
    """Integration tests for mooncake_trace dataset loader."""

    async def test_basic_mooncake_trace_with_input_length(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test basic Mooncake trace with input_length, output_length, and hash_ids."""
        # Real trace data from mooncake_trace.jsonl (first 5 lines)
        traces = [
            {"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
            {"timestamp": 0, "input_length": 7319, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
            {"timestamp": 0, "input_length": 7234, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]},
            {"timestamp": 0, "input_length": 2287, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]},
            {"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]},
        ]  # fmt: skip
        trace_file = create_mooncake_trace_file(tmp_path, traces)
        request_count = len(traces)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {trace_file} \
                --custom-dataset-type mooncake_trace \
                --request-count {request_count} \
                --fixed-schedule \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_mooncake_trace_with_text_input(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test Mooncake trace with literal text inputs instead of input_length."""
        # Each trace is a single-turn conversation; timestamp required for --fixed-schedule
        traces = [
            {"timestamp": 0, "text_input": "What is the capital of France?", "output_length": 20},
            {"timestamp": 100, "text_input": "Explain quantum computing briefly.", "output_length": 30},
            {"timestamp": 200, "text_input": "Write a haiku about programming.", "output_length": 25},
            {"timestamp": 300, "text_input": "What is machine learning?", "output_length": 40},
            {"timestamp": 400, "text_input": "Describe the solar system.", "output_length": 35},
        ]  # fmt: skip
        trace_file = create_mooncake_trace_file(tmp_path, traces)
        request_count = len(traces)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {trace_file} \
                --custom-dataset-type mooncake_trace \
                --request-count {request_count} \
                --fixed-schedule \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

    async def test_mooncake_trace_multi_turn_with_session_id(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test Mooncake trace with session_id for multi-turn conversations."""
        # First turn of each session needs timestamp; subsequent turns use delay
        traces = [
            # Session 1: Two-turn conversation (starts at t=0)
            {"session_id": "session-1", "timestamp": 0, "input_length": 100, "output_length": 40},
            {"session_id": "session-1", "delay": 500, "input_length": 150, "output_length": 50},
            # Session 2: Single-turn (starts at t=100)
            {"session_id": "session-2", "timestamp": 100, "input_length": 200, "output_length": 60},
            # Session 3: Three-turn conversation (starts at t=200)
            {"session_id": "session-3", "timestamp": 200, "input_length": 80, "output_length": 30},
            {"session_id": "session-3", "delay": 300, "input_length": 120, "output_length": 45},
            {"session_id": "session-3", "delay": 400, "input_length": 90, "output_length": 35},
        ]  # fmt: skip
        trace_file = create_mooncake_trace_file(tmp_path, traces)
        request_count = len(traces)  # Each turn is a request

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {trace_file} \
                --custom-dataset-type mooncake_trace \
                --request-count {request_count} \
                --fixed-schedule \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == request_count
        assert result.has_all_outputs

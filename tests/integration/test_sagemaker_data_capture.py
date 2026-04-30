# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for SageMaker Data Capture trace loader."""

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import (
    create_sagemaker_capture_file,
    create_sagemaker_capture_record,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestSageMakerDataCaptureIntegration:
    """Integration tests for sagemaker_data_capture dataset loader."""

    async def test_basic_capture_replay(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test replaying captured chat completion requests."""
        records = [
            create_sagemaker_capture_record(
                messages=[
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                max_tokens=50,
                inference_time="2026-04-29T00:00:00Z",
            ),
            create_sagemaker_capture_record(
                messages=[{"role": "user", "content": "Tell me about Python."}],
                max_tokens=80,
                inference_time="2026-04-29T00:00:02Z",
            ),
            create_sagemaker_capture_record(
                messages=[{"role": "user", "content": "What is machine learning?"}],
                max_tokens=60,
                inference_time="2026-04-29T00:00:04Z",
            ),
        ]
        capture_file = create_sagemaker_capture_file(tmp_path, records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {capture_file} \
                --custom-dataset-type sagemaker_data_capture \
                --request-count {len(records)} \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == len(records)
        assert result.has_all_outputs

    async def test_capture_with_system_message(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test replaying captures with system messages in the messages array."""
        records = [
            create_sagemaker_capture_record(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                max_tokens=30,
                inference_time="2026-04-29T00:00:00Z",
            ),
            create_sagemaker_capture_record(
                messages=[
                    {"role": "system", "content": "You are a coding expert."},
                    {"role": "user", "content": "Write a hello world in Python."},
                ],
                max_tokens=100,
                inference_time="2026-04-29T00:00:02Z",
            ),
        ]
        capture_file = create_sagemaker_capture_file(tmp_path, records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {capture_file} \
                --custom-dataset-type sagemaker_data_capture \
                --request-count {len(records)} \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == len(records)

    async def test_capture_directory_input(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test loading captures from a directory with hourly-partitioned files."""
        # Simulate SageMaker's hourly directory structure
        hour_00 = tmp_path / "2026" / "04" / "29" / "00"
        hour_01 = tmp_path / "2026" / "04" / "29" / "01"
        hour_00.mkdir(parents=True)
        hour_01.mkdir(parents=True)

        create_sagemaker_capture_file(
            hour_00,
            [
                create_sagemaker_capture_record(
                    messages=[{"role": "user", "content": "Request from hour 0"}],
                    inference_time="2026-04-29T00:00:00Z",
                ),
            ],
            filename="capture-00.jsonl",
        )
        create_sagemaker_capture_file(
            hour_01,
            [
                create_sagemaker_capture_record(
                    messages=[{"role": "user", "content": "Request from hour 1"}],
                    inference_time="2026-04-29T00:00:02Z",
                ),
            ],
            filename="capture-01.jsonl",
        )

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {tmp_path} \
                --custom-dataset-type sagemaker_data_capture \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == 2
        assert result.has_all_outputs

    async def test_capture_auto_detection(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test that the loader auto-detects SageMaker Data Capture format."""
        records = [
            create_sagemaker_capture_record(
                messages=[{"role": "user", "content": "Auto-detect test"}],
                inference_time="2026-04-29T00:00:00Z",
            ),
        ]
        capture_file = create_sagemaker_capture_file(tmp_path, records)

        # No --custom-dataset-type flag — should auto-detect
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {capture_file} \
                --request-count 1 \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == 1

    async def test_capture_with_base64_encoding(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ):
        """Test replaying captures with BASE64-encoded payloads."""
        records = [
            create_sagemaker_capture_record(
                messages=[{"role": "user", "content": "Base64 test request 1"}],
                max_tokens=40,
                inference_time="2026-04-29T00:00:00Z",
                encoding="BASE64",
            ),
            create_sagemaker_capture_record(
                messages=[{"role": "user", "content": "Base64 test request 2"}],
                max_tokens=60,
                inference_time="2026-04-29T00:00:02Z",
                encoding="BASE64",
            ),
        ]
        capture_file = create_sagemaker_capture_file(tmp_path, records)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {capture_file} \
                --custom-dataset-type sagemaker_data_capture \
                --request-count {len(records)} \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == len(records)
        assert result.has_all_outputs

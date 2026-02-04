# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU telemetry collection and reporting."""

import platform

import orjson
import pytest

from aiperf.common.models.telemetry_models import TelemetryRecord
from tests.harness.utils import AIPerfCLI, AIPerfMockServer


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Requires NVIDIA GPUs for DCGM telemetry (only available on Linux CI).",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestGpuTelemetry:
    """Tests for GPU telemetry collection and reporting."""

    async def test_gpu_telemetry(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """GPU telemetry collection with DCGM endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 100 \
                --concurrency 2 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        assert result.request_count == 100
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0

        for dcgm_url in result.json.telemetry_data.endpoints:
            assert result.json.telemetry_data.endpoints[dcgm_url].gpus is not None
            assert len(result.json.telemetry_data.endpoints[dcgm_url].gpus) > 0

            for gpu_data in result.json.telemetry_data.endpoints[
                dcgm_url
            ].gpus.values():
                assert gpu_data.metrics is not None
                assert len(gpu_data.metrics) > 0

                # Counter metrics only have avg (delta), not min/max
                counter_metrics = {
                    "energy_consumption",
                    "xid_errors",
                    "power_violation",
                }
                for metric_name, metric_value in gpu_data.metrics.items():
                    assert metric_value is not None
                    assert metric_value.avg is not None
                    assert metric_value.unit is not None
                    # Gauge metrics should have min/max; counter metrics only have avg
                    if metric_name not in counter_metrics:
                        assert metric_value.min is not None
                        assert metric_value.max is not None

    async def test_gpu_telemetry_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test GPU telemetry export to JSONL file with validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 50
        assert result.has_gpu_telemetry

        # Verify GPU telemetry export JSONL file exists
        export_file = result.artifacts_dir / "gpu_telemetry_export.jsonl"
        assert export_file.exists(), "GPU telemetry export file should exist"

        # Read and validate JSONL content
        content = export_file.read_text()
        lines = content.splitlines()
        assert len(lines) > 0, "Export file should contain telemetry records"

        # Collect GPU data for validation
        gpu_uuids = set()
        timestamps = []

        # Validate each line is valid JSON and can be parsed as TelemetryRecord
        for line in lines:
            record_dict = orjson.loads(line)
            record = TelemetryRecord.model_validate(record_dict)

            # Verify required fields are present
            assert record.timestamp_ns > 0
            assert record.dcgm_url is not None
            assert record.gpu_index >= 0
            assert record.gpu_uuid is not None
            assert record.gpu_model_name is not None
            assert record.telemetry_data is not None

            # Collect data for validation
            gpu_uuids.add(record.gpu_uuid)
            timestamps.append(record.timestamp_ns)

        # Verify we captured data from GPUs
        assert len(gpu_uuids) >= 2, "Should have records from at least two GPUs"

        # NOTE: Records are not necessarily in timestamp order because of the asynchronous
        # nature of the telemetry collection.

    async def test_gpu_telemetry_export_with_custom_prefix(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test GPU telemetry export with custom filename prefix."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --profile-export-prefix custom_test
            """
        )

        # Verify custom filename is used
        export_file = result.artifacts_dir / "custom_test_gpu_telemetry.jsonl"
        if export_file.exists():
            # Verify content is valid
            content = export_file.read_text()
            lines = content.splitlines()
            assert len(lines) > 0, "Export file should contain telemetry records"

            # Validate first record
            first_record = TelemetryRecord.model_validate_json(lines[0])
            assert first_record.timestamp_ns > 0
            assert first_record.dcgm_url is not None

    async def test_gpu_telemetry_disabled(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """GPU telemetry collection is disabled with --no-gpu-telemetry flag.

        When --no-gpu-telemetry is provided, no GPU telemetry files should be
        created and no telemetry should be collected, even if DCGM endpoints
        would otherwise be reachable.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --no-gpu-telemetry
            """
        )
        assert result.request_count == 25

        # GPU telemetry should NOT be collected when disabled
        assert not result.has_gpu_telemetry, "GPU telemetry should not be collected"

        # Verify no GPU telemetry files were created
        jsonl_files = list(result.artifacts_dir.glob("*gpu_telemetry*.jsonl"))
        assert len(jsonl_files) == 0, f"Unexpected GPU telemetry files: {jsonl_files}"

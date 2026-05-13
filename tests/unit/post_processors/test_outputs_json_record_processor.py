# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OutputsJsonRecordProcessor."""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, OutputConfig, UserConfig
from aiperf.common.enums import ExportLevel
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models.record_models import (
    MetricRecordMetadata,
    ParsedResponseRecord,
)
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.outputs_json_record_processor import (
    OutputsJsonRecordProcessor,
)
from tests.unit.post_processors.conftest import aiperf_lifecycle


class TestOutputsJsonRecordProcessorDisabled:
    """Tests for OutputsJsonRecordProcessor disabled state."""

    def test_disabled_when_flag_not_set(self, tmp_path: Path) -> None:
        """Raises PostProcessorDisabled when export_outputs_json is False."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            ),
            output=OutputConfig(
                artifact_directory=tmp_path,
                export_outputs_json=False,
            ),
        )

        with pytest.raises(PostProcessorDisabled):
            OutputsJsonRecordProcessor(
                service_id="processor-1",
                user_config=config,
            )


class TestOutputsJsonRecordProcessorProcessRecord:
    """Tests for OutputsJsonRecordProcessor process_record method."""

    @pytest.mark.asyncio
    async def test_process_record_writes_fragment(self, tmp_path: Path) -> None:
        """Creates a mock ParsedResponseRecord with content_responses, calls process_record, verifies fragment is written."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            ),
            output=OutputConfig(
                artifact_directory=tmp_path,
                export_outputs_json=True,
                export_level=ExportLevel.RECORDS,
            ),
        )

        record = MagicMock(spec=ParsedResponseRecord)
        resp1 = MagicMock()
        resp1.data.get_text.return_value = "Hello "
        resp2 = MagicMock()
        resp2.data.get_text.return_value = "world!"
        type(record).content_responses = PropertyMock(return_value=[resp1, resp2])

        metadata = MetricRecordMetadata(
            session_num=0,
            request_start_ns=1000000000,
            request_end_ns=2000000000,
            worker_id="worker-1",
            record_processor_id="proc-1",
            benchmark_phase="profiling",
        )

        processor = OutputsJsonRecordProcessor(
            service_id="processor-1",
            user_config=config,
        )
        async with aiperf_lifecycle(processor) as proc:
            await proc.process_record(record, metadata)

        assert proc.lines_written == 1

    @pytest.mark.asyncio
    async def test_process_record_extracts_response_text(self, tmp_path: Path) -> None:
        """Verifies response text is concatenated from content_responses."""
        import orjson

        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            ),
            output=OutputConfig(
                artifact_directory=tmp_path,
                export_outputs_json=True,
                export_level=ExportLevel.RECORDS,
            ),
        )

        record = MagicMock(spec=ParsedResponseRecord)
        resp1 = MagicMock()
        resp1.data.get_text.return_value = "Hello "
        resp2 = MagicMock()
        resp2.data.get_text.return_value = "world!"
        type(record).content_responses = PropertyMock(return_value=[resp1, resp2])

        metadata = MetricRecordMetadata(
            session_num=0,
            request_start_ns=1000000000,
            request_end_ns=2000000000,
            worker_id="worker-1",
            record_processor_id="proc-1",
            benchmark_phase="profiling",
        )

        processor = OutputsJsonRecordProcessor(
            service_id="processor-1",
            user_config=config,
        )
        async with aiperf_lifecycle(processor) as proc:
            await proc.process_record(record, metadata)

        # Read the written fragment file and verify response_text
        output_file = proc.output_file
        content = output_file.read_bytes()
        fragment = orjson.loads(content.strip())
        assert fragment["response_text"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_process_record_null_response_text_when_no_content(
        self, tmp_path: Path
    ) -> None:
        """When content_responses is empty, response_text is None."""
        import orjson

        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            ),
            output=OutputConfig(
                artifact_directory=tmp_path,
                export_outputs_json=True,
                export_level=ExportLevel.RECORDS,
            ),
        )

        record = MagicMock(spec=ParsedResponseRecord)
        type(record).content_responses = PropertyMock(return_value=[])

        metadata = MetricRecordMetadata(
            session_num=0,
            request_start_ns=1000000000,
            request_end_ns=2000000000,
            worker_id="worker-1",
            record_processor_id="proc-1",
            benchmark_phase="profiling",
        )

        processor = OutputsJsonRecordProcessor(
            service_id="processor-1",
            user_config=config,
        )
        async with aiperf_lifecycle(processor) as proc:
            await proc.process_record(record, metadata)

        # Read the written fragment file and verify response_text is absent (exclude_none=True)
        output_file = proc.output_file
        content = output_file.read_bytes()
        fragment = orjson.loads(content.strip())
        assert "response_text" not in fragment


class TestOutputConfigExportOutputsJsonValidation:
    """Tests for OutputConfig validation of export_outputs_json."""

    def test_export_outputs_json_with_summary_raises(self) -> None:
        """export_outputs_json=True with export_level=summary raises ValueError."""
        with pytest.raises(ValidationError, match="export-outputs-json"):
            OutputConfig(
                export_outputs_json=True,
                export_level=ExportLevel.SUMMARY,
            )

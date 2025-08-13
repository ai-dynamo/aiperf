# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import DataExporterType, EndpointType
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_manager import ExporterManager


@pytest.fixture
def endpoint_config():
    return EndpointConfig(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS, streaming=True, model_names=["gpt2"]
    )


@pytest.fixture
def output_config(tmp_path):
    return OutputConfig(artifact_directory=tmp_path)


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="Latency",
            unit="ms",
            avg=10.0,
            header="test-header",
        )
    ]


@pytest.fixture
def mock_user_config(endpoint_config, output_config):
    config = UserConfig(endpoint=endpoint_config, output=output_config)
    return config


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        exporter_types = [
            DataExporterType.JSON,
        ]
        mock_exporter_instances = []
        mock_exporter_classes = {}

        for exporter_type in exporter_types:
            instance = MagicMock()
            instance.export = AsyncMock()
            mock_class = MagicMock(return_value=instance)
            mock_exporter_classes[exporter_type] = mock_class
            mock_exporter_instances.append(instance)

        with patch.object(
            __import__(
                "aiperf.common.factories", fromlist=["DataExporterFactory"]
            ).DataExporterFactory,
            "_registry",
            mock_exporter_classes,
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                input_config=mock_user_config,
                service_config=ServiceConfig(),
            )
            await manager.export_data()
        for mock_class, mock_instance in zip(
            mock_exporter_classes.values(), mock_exporter_instances, strict=False
        ):
            mock_class.assert_called_once()
            mock_instance.export.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_console(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        exporter_types = [
            ConsoleExporterType.METRICS,
            ConsoleExporterType.ERRORS,
        ]
        mock_exporter_instances = []
        mock_exporter_classes = {}
        from rich.console import Console

        for exporter_type in exporter_types:
            instance = MagicMock()
            instance.export = AsyncMock()
            mock_class = MagicMock(return_value=instance)
            mock_exporter_classes[exporter_type] = mock_class
            mock_exporter_instances.append(instance)

        with patch.object(
            __import__(
                "aiperf.common.factories", fromlist=["ConsoleExporterFactory"]
            ).ConsoleExporterFactory,
            "_registry",
            mock_exporter_classes,
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                input_config=mock_user_config,
                service_config=ServiceConfig(),
            )
            await manager.export_console(Console())
        for mock_class, mock_instance in zip(
            mock_exporter_classes.values(), mock_exporter_instances, strict=False
        ):
            mock_class.assert_called_once()
            mock_instance.export.assert_awaited_once()

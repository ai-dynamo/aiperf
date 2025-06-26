# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.table import Table

from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.data_exporter.exporter_config import ExporterConfig
from aiperf.data_exporter.record import Record


@DataExporterFactory.register(DataExporterType.CONSOLE)
class ConsoleExporter:
    """A class that exports data to the console"""

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p75"]

    def __init__(self, exporter_config: ExporterConfig) -> None:
        self._records = exporter_config.records
        self._endpoint_type = exporter_config.input_config.endpoint.type
        self._streaming = exporter_config.input_config.endpoint.streaming

    def export(self, **kwargs) -> None:
        console = Console(**kwargs)
        table = Table(title=self._get_title())
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.STAT_COLUMN_KEYS:
            table.add_column(key, justify="right", style="green")
        self._construct_table(table)
        console.print(table)

    def _construct_table(self, table: Table) -> None:
        for record in self._records:
            if self._should_skip(record):
                continue
            table.add_row(*self._format_row(record))

    def _should_skip(self, record: Record) -> bool:
        if self._endpoint_type == "embeddings":
            return False

        return record.streaming_only and not self._streaming

    def _format_row(self, record: Record) -> list[str]:
        row = [f"{record.name} ({record.unit})"]
        for stat in self.STAT_COLUMN_KEYS:
            value = getattr(record, stat, None)
            row.append(f"{value:.2f}" if value is not None else "N/A")
        return row

    def _get_title(self) -> str:
        type_titles = {
            "embeddings": "Embeddings Metrics",
            "rankings": "Rankings Metrics",
            "image_retrieval": "Image Retrieval Metrics",
            "multimodal": "Multi-Modal Metrics",
        }
        metric_title = type_titles.get(self._endpoint_type, "LLM Metrics")
        return f"NVIDIA AIPerf | {metric_title}"

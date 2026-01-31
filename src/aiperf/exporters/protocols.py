# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console

    from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@runtime_checkable
class ConsoleExporterProtocol(Protocol):
    """Protocol for console exporters.
    Any class implementing this protocol will be provided an ExporterConfig and must provide an
    `export` method that takes a rich Console and handles exporting them appropriately.
    """

    def __init__(self, exporter_config: ExporterConfig) -> None: ...

    async def export(self, console: Console) -> None: ...


@runtime_checkable
class DataExporterProtocol(Protocol):
    """
    Protocol for data exporters.
    Any class implementing this protocol will be provided an ExporterConfig and must provide an
    `export` method that handles exporting the data appropriately.
    """

    def __init__(self, exporter_config: ExporterConfig) -> None: ...

    def get_export_info(self) -> FileExportInfo: ...

    async def export(self) -> None: ...

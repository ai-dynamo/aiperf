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


@runtime_checkable
class ArtifactPublisherProtocol(Protocol):
    """Protocol for artifact publishers that upload exported files to remote storage.

    Artifact publishers run after all data and stream exporters have completed.
    They receive the full list of exported file paths and upload them to remote
    storage backends (S3, GCS, Azure Blob, etc.).
    """

    def __init__(self, exporter_config: ExporterConfig) -> None: ...

    async def publish(self, artifacts: list[FileExportInfo]) -> None:
        """Upload artifacts to remote storage.

        Args:
            artifacts: File paths and their types from all exporters.
                       Publishers may filter by export_type or publish all.
        """
        ...

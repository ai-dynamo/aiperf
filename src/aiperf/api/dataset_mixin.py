# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset mixin for the API service. Provides the dataset client metadata and configured event.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import MemoryMapClientMetadata


class DatasetMixin:
    """Dataset mixin for the API service."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mixin."""
        super().__init__(**kwargs)

        # Dataset metadata received from DatasetManager
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._benchmark_generation: str | None = None
        self._dataset_generation: str | None = None
        self._dataset_configured = asyncio.Event()

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Handle dataset configuration notification from DatasetManager.

        Stores the client metadata containing file paths for the dataset endpoints.
        This ensures the API service knows exactly where the dataset files are,
        including whether compressed files exist.
        """
        if isinstance(message.client_metadata, MemoryMapClientMetadata):
            self._dataset_client_metadata = message.client_metadata
            self._benchmark_generation = message.benchmark_generation
            self._dataset_generation = message.dataset_generation
            self._dataset_configured.set()
            self.info(
                f"Dataset configured: {message.client_metadata.conversation_count} conversations, "
                f"compressed={message.client_metadata.compressed}"
            )
        else:
            self.warning(
                f"Received dataset metadata with unsupported type: {type(message.client_metadata)}"
            )

    @property
    def dataset_client_metadata(self) -> MemoryMapClientMetadata | None:
        """Get the dataset client metadata."""
        return self._dataset_client_metadata

    @property
    def benchmark_generation(self) -> str | None:
        """Get the active benchmark generation."""
        return self._benchmark_generation

    @property
    def dataset_generation(self) -> str | None:
        """Get the active dataset generation."""
        return self._dataset_generation

    @property
    def dataset_configured(self) -> asyncio.Event:
        """Get the dataset configured event."""
        return self._dataset_configured

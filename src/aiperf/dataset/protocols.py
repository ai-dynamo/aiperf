# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aiperf.common.models import Conversation
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.models.dataset_models import DatasetClientMetadata
    from aiperf.dataset.loader.models import CustomDatasetT
    from aiperf.plugin.enums import DatasetSamplingStrategy


@runtime_checkable
class CustomDatasetLoaderProtocol(Protocol):
    """Protocol for custom dataset loaders that load dataset from a file and convert it to a list of Conversation objects."""

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: "str | Path | None" = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        Args:
            data: Optional dictionary representing a single line from the JSONL file.
                  None indicates path-based detection only (e.g., for directories).
            filename: Optional path to the input file/directory for path-based detection

        Returns:
            True if this loader can handle the data format, False otherwise
        """
        ...

    @classmethod
    def get_preferred_sampling_strategy(cls) -> "DatasetSamplingStrategy":
        """Get the preferred dataset sampling strategy for this loader.

        Returns:
            DatasetSamplingStrategy: The preferred sampling strategy
        """
        ...

    def load_dataset(self) -> dict[str, list["CustomDatasetT"]]: ...

    def convert_to_conversations(
        self, custom_data: dict[str, list["CustomDatasetT"]]
    ) -> list[Conversation]: ...


@runtime_checkable
class DatasetSamplingStrategyProtocol(Protocol):
    """
    Protocol for dataset sampling strategies.
    Any class implementing this protocol will be provided a list of conversation ids, and must
    provide a `next_conversation_id` method that returns the next conversation id, ensuring reproducibility.
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None: ...
    def next_conversation_id(self) -> str: ...


@runtime_checkable
class DatasetBackingStoreProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for creating and managing dataset storage (DatasetManager side).

    Extends AIPerfLifecycleProtocol for lifecycle management, logging, and task management.

    **Usage Pattern**:
    1. Create backing store with __init__
    2. Call initialize()
    3. Add conversations with add_conversation() or add_conversations()
    4. Call finalize() to make data available to clients

    **ORDER GUARANTEE**: Conversation insertion order MUST be preserved.
    This is critical for:
    - Datasets with timing data (fixed schedule)
    - Reproducibility
    - Sequential sampling strategies

    Implementations MUST maintain insertion order using dict (Python 3.7+) or OrderedDict.
    All implementations MUST support the streaming API.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize backing store.

        Args:
            **kwargs: Implementation-specific configuration
                     (e.g., shared_mount_path, redis_url)
        """
        ...

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation (streaming mode).

        Conversations are added in the order this method is called.
        Order MUST be preserved for sequential access and timing data.

        Args:
            conversation_id: Session ID of the conversation
            conversation: Conversation object to add

        Raises:
            RuntimeError: If store not initialized or already finalized
        """
        ...

    async def add_conversations(self, conversations: dict[str, Conversation]) -> None:
        """Add multiple conversations (streaming mode).

        More efficient than individual add_conversation() calls.
        Insertion order from the dict MUST be preserved.

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects.
                          Dict insertion order is preserved (Python 3.7+).

        Raises:
            RuntimeError: If store not initialized or already finalized
        """
        ...

    async def finalize(self) -> None:
        """Finalize streaming and make data available to clients.

        Call after all add_conversation/add_conversations calls.
        Creates indexes, flushes buffers, closes files, etc.

        Raises:
            RuntimeError: If store not initialized
        """
        ...

    def get_client_metadata(self) -> "DatasetClientMetadata":
        """Get metadata needed by clients to connect to this storage.

        Returns:
            DatasetClientMetadata subclass with connection information

        Raises:
            RuntimeError: If additions not finalized (streaming mode)
        """
        ...


@runtime_checkable
class DatasetClientStoreProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for accessing dataset storage (Worker side).

    Extends AIPerfLifecycleProtocol for lifecycle management, logging, and task management.
    """

    def __init__(self, client_metadata: "DatasetClientMetadata", **kwargs) -> None:
        """Initialize client store from backing store metadata.

        Args:
            client_metadata: Typed metadata from DatasetBackingStore.get_client_metadata()
            **kwargs: Additional client-specific configuration
        """
        ...

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve a conversation by ID (always async).

        Args:
            conversation_id: The session ID of the conversation

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id not found
        """
        ...

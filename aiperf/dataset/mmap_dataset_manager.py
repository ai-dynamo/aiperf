# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import json
import mmap
import os
import random
import tempfile
import weakref
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import aiofiles
from pydantic import BaseModel, Field, field_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.decorators import implements_protocol
from aiperf.common.models import Conversation
from aiperf.common.protocols import ServiceProtocol

_logger = AIPerfLogger(__name__)


class MMapDatasetError(Exception):
    """Base exception for memory-mapped dataset operations."""

    pass


class MMapSerializationError(MMapDatasetError):
    """Exception raised during serialization/deserialization operations."""

    pass


class MMapFileOperationError(MMapDatasetError):
    """Exception raised during file operations."""

    pass


class MMapResourceError(MMapDatasetError):
    """Exception raised during resource management operations."""

    pass


class MMapFileType(str, Enum):
    """Enumeration for memory-mapped file types."""

    DATA = "data"
    INDEX = "index"


class SerializationFormat(str, Enum):
    """Enumeration for serialization formats."""

    JSON = "json"
    JSON_COMPRESSED = "json_gzip"


@dataclass(frozen=True)
class MMapConstants:
    """Constants for memory-mapped dataset operations."""

    TEMP_DIR_NAME: Final[str] = "aiperf_mmap"
    DATA_FILE_SUFFIX: Final[str] = ".dat"
    INDEX_FILE_SUFFIX: Final[str] = ".dat"
    ALL_CONVERSATIONS_KEY: Final[str] = "__all__"
    JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
    DEFAULT_COMPRESSION_LEVEL: Final[int] = 6


# Global instance for easy access
MMAP_CONSTANTS = MMapConstants()


class MMapDatasetIndex(BaseModel):
    """Index structure for the memory-mapped dataset."""

    conversations: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Mapping of session_id to (offset, length) tuples",
    )
    session_ids: list[str] = Field(
        default_factory=list, description="List of all session IDs in the dataset"
    )
    total_size: int = Field(
        default=0, ge=0, description="Total size of the serialized dataset in bytes"
    )
    use_sequential_iteration: bool = Field(
        default=False,
        description="Whether to use sequential iteration instead of random",
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducible random access"
    )
    serialization_format: SerializationFormat = Field(
        default=SerializationFormat.JSON, description="Format used for serialization"
    )

    @field_validator("session_ids")
    @classmethod
    def validate_session_ids(cls, v: list[str]) -> list[str]:
        """Validate that session_ids contains unique strings."""
        if len(v) != len(set(v)):
            raise ValueError("session_ids must contain unique values")
        return v


class MMapDatasetSerializer:
    """Utilities for serializing and deserializing dataset objects for memory mapping."""

    @staticmethod
    def serialize_conversations(
        conversations: dict[str, Conversation], compression: bool = False
    ) -> tuple[bytes, SerializationFormat]:
        """Serialize conversations to bytes for memory mapping.

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects
            compression: Whether to use gzip compression

        Returns:
            Tuple of (serialized data bytes, format used)

        Raises:
            ValueError: If conversations dict is empty
            TypeError: If conversation objects are invalid
        """
        if not conversations:
            raise MMapSerializationError(
                "Cannot serialize empty conversations dictionary"
            )

        try:
            serialized_data = {
                session_id: conversation.model_dump(mode="json")
                for session_id, conversation in conversations.items()
            }

            json_str = json.dumps(
                serialized_data,
                separators=MMAP_CONSTANTS.JSON_SEPARATORS,
                ensure_ascii=False,
            )
            json_bytes = json_str.encode("utf-8")

            if compression:
                compressed_data = gzip.compress(
                    json_bytes, compresslevel=MMAP_CONSTANTS.DEFAULT_COMPRESSION_LEVEL
                )
                return compressed_data, SerializationFormat.JSON_COMPRESSED
            else:
                return json_bytes, SerializationFormat.JSON

        except (TypeError, AttributeError) as e:
            raise MMapSerializationError(
                f"Invalid conversation object during serialization: {e}"
            ) from e
        except UnicodeEncodeError as e:
            raise MMapSerializationError(
                f"Failed to encode conversation data: {e}"
            ) from e

    @staticmethod
    def deserialize_conversations(
        data: bytes, format_type: SerializationFormat = SerializationFormat.JSON
    ) -> dict[str, Conversation]:
        """Deserialize conversations from bytes stored in memory-mapped file.

        Args:
            data: Serialized conversation data bytes
            format_type: Format of the serialized data

        Returns:
            Dictionary mapping session IDs to Conversation objects

        Raises:
            ValueError: If data cannot be decoded or parsed as JSON
            TypeError: If conversation data is invalid
        """
        try:
            if format_type == SerializationFormat.JSON_COMPRESSED:
                # Decompress gzip data first
                json_bytes = gzip.decompress(data)
                json_str = json_bytes.decode("utf-8")
            else:
                # Regular JSON format
                json_str = data.decode("utf-8")

            serialized_data = json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError, gzip.BadGzipFile) as e:
            raise MMapSerializationError(
                f"Failed to decode conversation data ({format_type.value}): {e}"
            ) from e

        try:
            return {
                session_id: Conversation.model_validate(conv_data)
                for session_id, conv_data in serialized_data.items()
            }
        except Exception as e:
            raise MMapSerializationError(
                f"Failed to validate conversation data: {e}"
            ) from e

    @staticmethod
    def create_index(
        conversations: dict[str, Conversation],
        serialized_data: bytes,
        format_type: SerializationFormat = SerializationFormat.JSON,
    ) -> MMapDatasetIndex:
        """Create an index for efficient access to conversations in memory-mapped file.

        Args:
            conversations: Dictionary of conversations to index
            serialized_data: Serialized conversation data
            format_type: Format used for serialization

        Returns:
            MMapDatasetIndex with conversation metadata

        Raises:
            ValueError: If conversations is empty or serialized_data is invalid
        """
        if not conversations:
            raise MMapSerializationError("Cannot create index for empty conversations")
        if not serialized_data:
            raise MMapSerializationError(
                "Cannot create index for empty serialized data"
            )

        return MMapDatasetIndex(
            session_ids=list(conversations.keys()),
            total_size=len(serialized_data),
            conversations={
                MMAP_CONSTANTS.ALL_CONVERSATIONS_KEY: (0, len(serialized_data))
            },
            serialization_format=format_type,
        )


@implements_protocol(ServiceProtocol)
class MMapDatasetManager:
    """
    Manages a memory-mapped dataset that can be accessed by multiple worker processes.

    This class creates and manages memory-mapped files containing the dataset,
    allowing workers to access conversation data without making network requests
    to the dataset manager. This provides robust cross-platform memory mapping.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(
        self,
        dataset: dict[str, Conversation],
        random_seed: int | None = None,
        use_sequential_iteration: bool = False,
        enable_compression: bool = False,
    ) -> None:
        """Initialize the MMapDatasetManager.

        Args:
            dataset: Dictionary mapping session IDs to Conversation objects
            random_seed: Optional random seed for reproducible random access
            use_sequential_iteration: Whether to use sequential instead of random iteration
            enable_compression: Whether to enable gzip compression for data storage

        Raises:
            ValueError: If dataset is empty
            TypeError: If dataset contains invalid conversation objects
        """
        if not dataset:
            raise MMapDatasetError("Dataset cannot be empty")

        self.dataset = dataset
        self.random_seed = random_seed
        self.use_sequential_iteration = use_sequential_iteration
        self.enable_compression = enable_compression

        # Serialize the dataset with optional compression
        self.serialized_data, self.serialization_format = (
            MMapDatasetSerializer.serialize_conversations(
                dataset, compression=enable_compression
            )
        )
        self.index = MMapDatasetSerializer.create_index(
            dataset, self.serialized_data, self.serialization_format
        )
        self.index.random_seed = random_seed
        self.index.use_sequential_iteration = use_sequential_iteration

        # Memory-mapped file paths
        self.data_file_path: Path | None = None
        self.index_file_path: Path | None = None
        self.data_mmap: mmap.mmap | None = None
        self.index_mmap: mmap.mmap | None = None
        self._files_created = False

        self._finalizer = weakref.finalize(
            self, self._cleanup_finalizer, self.data_file_path, self.index_file_path
        )

        _logger.debug(
            "MMapDatasetManager initialized successfully",
            extra={
                "component": "MMapDatasetManager",
                "conversations_count": len(dataset),
                "random_seed": random_seed,
                "sequential_iteration": use_sequential_iteration,
                "compression_enabled": enable_compression,
                "serialization_format": self.serialization_format.value,
                "serialized_size_bytes": len(self.serialized_data),
                "dataset_keys_sample": list(dataset.keys())[:5] if dataset else [],
            },
        )

    def __enter__(self) -> "MMapDatasetManager":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup_memory_mapped_files()

    async def __aenter__(self) -> "MMapDatasetManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Async context manager exit with automatic cleanup."""
        await self.cleanup_memory_mapped_files_async()

    @staticmethod
    def _cleanup_finalizer(
        data_file_path: Path | None, index_file_path: Path | None
    ) -> None:
        """Cleanup method called during garbage collection."""
        for file_path in [data_file_path, index_file_path]:
            if file_path and file_path.exists():
                with suppress(OSError):
                    file_path.unlink()
                    _logger.debug(f"Finalizer cleaned up file: {file_path}")

    def create_memory_mapped_files(self) -> tuple[str, str]:
        """Create memory-mapped files and return their paths.

        Returns:
            Tuple of (data_file_path, index_file_path) as strings

        Raises:
            OSError: If file creation fails
            PermissionError: If insufficient permissions for file creation
        """
        if self._files_created:
            _logger.warning("Memory-mapped files already created")
            return str(self.data_file_path), str(self.index_file_path)

        try:
            # Create temporary directory for memory-mapped files
            temp_dir = Path(tempfile.gettempdir()) / MMAP_CONSTANTS.TEMP_DIR_NAME
            temp_dir.mkdir(exist_ok=True)

            # Generate unique file names using PID and object ID
            unique_suffix = f"{os.getpid()}_{id(self)}"
            self.data_file_path = (
                temp_dir / f"dataset_{unique_suffix}{MMAP_CONSTANTS.DATA_FILE_SUFFIX}"
            )
            self.index_file_path = (
                temp_dir / f"index_{unique_suffix}{MMAP_CONSTANTS.INDEX_FILE_SUFFIX}"
            )

            # Create data file
            self.data_file_path.write_bytes(self.serialized_data)

            # Create index file with Pydantic model serialization
            index_data = self.index.model_dump_json(by_alias=True).encode("utf-8")
            self.index_file_path.write_bytes(index_data)

            self._files_created = True
            _logger.info(
                "Memory-mapped files created successfully",
                extra={
                    "component": "MMapDatasetManager",
                    "operation": "create_memory_mapped_files",
                    "data_file_path": str(self.data_file_path),
                    "index_file_path": str(self.index_file_path),
                    "data_file_size_bytes": self.data_file_path.stat().st_size,
                    "index_file_size_bytes": self.index_file_path.stat().st_size,
                    "serialization_format": self.serialization_format.value,
                    "conversations_count": len(self.index.session_ids),
                },
            )
            return str(self.data_file_path), str(self.index_file_path)

        except (OSError, PermissionError) as e:
            _logger.error(f"Failed to create memory-mapped files: {e}")
            # Clean up any partially created files
            self._cleanup_partial_files()
            raise MMapFileOperationError(
                f"Failed to create memory-mapped files: {e}"
            ) from e

    async def create_memory_mapped_files_async(self) -> tuple[str, str]:
        """Async version of create_memory_mapped_files for better I/O performance.

        Returns:
            Tuple of (data_file_path, index_file_path) as strings

        Raises:
            OSError: If file creation fails
            PermissionError: If insufficient permissions for file creation
        """
        if self._files_created:
            _logger.warning("Memory-mapped files already created")
            return str(self.data_file_path), str(self.index_file_path)

        try:
            # Create temporary directory for memory-mapped files
            temp_dir = Path(tempfile.gettempdir()) / MMAP_CONSTANTS.TEMP_DIR_NAME
            temp_dir.mkdir(exist_ok=True)

            # Generate unique file names using PID and object ID
            unique_suffix = f"{os.getpid()}_{id(self)}"
            self.data_file_path = (
                temp_dir / f"dataset_{unique_suffix}{MMAP_CONSTANTS.DATA_FILE_SUFFIX}"
            )
            self.index_file_path = (
                temp_dir / f"index_{unique_suffix}{MMAP_CONSTANTS.INDEX_FILE_SUFFIX}"
            )

            # Create data file asynchronously
            async with aiofiles.open(self.data_file_path, "wb") as f:
                await f.write(self.serialized_data)

            # Create index file with Pydantic model serialization
            index_data = self.index.model_dump_json(by_alias=True).encode("utf-8")
            async with aiofiles.open(self.index_file_path, "wb") as f:
                await f.write(index_data)

            self._files_created = True
            _logger.info(
                "Memory-mapped files created successfully (async)",
                extra={
                    "component": "MMapDatasetManager",
                    "operation": "create_memory_mapped_files_async",
                    "data_file_path": str(self.data_file_path),
                    "index_file_path": str(self.index_file_path),
                    "data_file_size_bytes": self.data_file_path.stat().st_size,
                    "index_file_size_bytes": self.index_file_path.stat().st_size,
                    "serialization_format": self.serialization_format.value,
                    "conversations_count": len(self.index.session_ids),
                },
            )
            return str(self.data_file_path), str(self.index_file_path)

        except (OSError, PermissionError) as e:
            _logger.error(f"Failed to create memory-mapped files (async): {e}")
            # Clean up any partially created files
            await self._cleanup_partial_files_async()
            raise MMapFileOperationError(
                f"Failed to create memory-mapped files (async): {e}"
            ) from e

    def _cleanup_partial_files(self) -> None:
        """Clean up any partially created files during error recovery."""
        for file_path in [self.data_file_path, self.index_file_path]:
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except OSError as e:
                    _logger.warning(f"Failed to remove partial file {file_path}: {e}")

    async def _cleanup_partial_files_async(self) -> None:
        """Async version of cleanup_partial_files."""
        tasks = []
        for file_path in [self.data_file_path, self.index_file_path]:
            if file_path and file_path.exists():
                task = asyncio.create_task(self._remove_file_async(file_path))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _remove_file_async(self, file_path: Path) -> None:
        """Asynchronously remove a file."""
        try:
            # Run the blocking unlink operation in a thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, file_path.unlink)
        except OSError as e:
            _logger.warning(f"Failed to remove partial file {file_path}: {e}")

    def cleanup_memory_mapped_files(self) -> None:
        """Clean up memory-mapped files and associated resources.

        This method safely closes any open memory maps and removes temporary files.
        It's safe to call multiple times.
        """
        if not self._files_created:
            return

        resources_to_close = [
            (self.data_mmap, "data_mmap", MMapFileType.DATA),
            (self.index_mmap, "index_mmap", MMapFileType.INDEX),
        ]

        for resource, attr_name, file_type in resources_to_close:
            if resource:
                try:
                    resource.close()
                    _logger.debug(f"Closed {file_type.value} {attr_name}")
                except Exception as e:
                    _logger.warning(f"Error closing {file_type.value} {attr_name}: {e}")
                finally:
                    setattr(self, attr_name, None)

        files_to_remove = [
            (self.data_file_path, MMapFileType.DATA),
            (self.index_file_path, MMapFileType.INDEX),
        ]

        for file_path, file_type in files_to_remove:
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                    _logger.debug(f"Removed {file_type.value} file: {file_path}")
                except OSError as e:
                    _logger.warning(
                        f"Error removing {file_type.value} file {file_path}: {e}"
                    )

        self.data_file_path = None
        self.index_file_path = None
        self._files_created = False

    async def cleanup_memory_mapped_files_async(self) -> None:
        """Async version of cleanup_memory_mapped_files for better I/O performance.

        This method safely closes any open memory maps and removes temporary files.
        It's safe to call multiple times.
        """
        if not self._files_created:
            return

        for mmap_obj, file_type in [
            (self.data_mmap, MMapFileType.DATA),
            (self.index_mmap, MMapFileType.INDEX),
        ]:
            if mmap_obj:
                try:
                    mmap_obj.close()
                except Exception as e:
                    _logger.warning(f"Error closing {file_type} mmap: {e}")

        self.data_mmap = None
        self.index_mmap = None

        removal_tasks = []
        for file_path, file_type in [
            (self.data_file_path, MMapFileType.DATA),
            (self.index_file_path, MMapFileType.INDEX),
        ]:
            if file_path and file_path.exists():
                task = asyncio.create_task(
                    self._remove_file_with_logging_async(file_path, file_type)
                )
                removal_tasks.append(task)

        if removal_tasks:
            await asyncio.gather(*removal_tasks, return_exceptions=True)

        self.data_file_path = None
        self.index_file_path = None
        self._files_created = False

    async def _remove_file_with_logging_async(
        self, file_path: Path, file_type: MMapFileType
    ) -> None:
        """Asynchronously remove a file with logging."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, file_path.unlink)
            _logger.debug(f"Removed {file_type.value} file: {file_path}")
        except OSError as e:
            _logger.warning(f"Error removing {file_type.value} file {file_path}: {e}")


class MMapDatasetClient:
    """
    Client for accessing memory-mapped dataset from worker processes.

    This class provides an interface for workers to access conversation data
    from memory-mapped files without making network requests.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(self, data_file_path: str, index_file_path: str) -> None:
        """Initialize the MMapDatasetClient.

        Args:
            data_file_path: Path to the memory-mapped data file
            index_file_path: Path to the memory-mapped index file

        Raises:
            FileNotFoundError: If either file doesn't exist
            OSError: If files cannot be opened or mapped
            ValueError: If index data is invalid
        """
        self.data_file_path = Path(data_file_path)
        self.index_file_path = Path(index_file_path)

        # Validate file existence
        if not self.data_file_path.exists():
            raise MMapFileOperationError(f"Data file not found: {data_file_path}")
        if not self.index_file_path.exists():
            raise MMapFileOperationError(f"Index file not found: {index_file_path}")

        try:
            # Open and map the files
            self.data_file = self.data_file_path.open("rb")
            self.data_mmap = mmap.mmap(
                self.data_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            self.index_file = self.index_file_path.open("rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            # Load and validate the index using Pydantic
            index_data = self.index_mmap.read()
            self.index = MMapDatasetIndex.model_validate_json(index_data)

        except OSError as e:
            # Clean up any partially opened resources
            self._cleanup_resources()
            raise MMapFileOperationError(
                f"Failed to open memory-mapped files: {e}"
            ) from e
        except (ValueError, json.JSONDecodeError) as e:
            # Clean up resources and re-raise as ValueError
            self._cleanup_resources()
            raise MMapSerializationError(f"Invalid index data: {e}") from e

        # Initialize random generator if needed
        self._conversation_query_random: random.Random | None = None
        if self.index.random_seed is not None:
            self._conversation_query_random = random.Random(self.index.random_seed)

        # Sequential iteration state
        self._sequential_iterator_index = 0

        # Cache for deserialized conversations
        self._conversations_cache: dict[str, Conversation] | None = None

        self._finalizer = weakref.finalize(
            self,
            self._cleanup_finalizer,
            self.data_mmap,
            self.index_mmap,
            self.data_file,
            self.index_file,
        )

        _logger.debug(
            "MMapDatasetClient initialized successfully",
            extra={
                "component": "MMapDatasetClient",
                "data_file_path": str(self.data_file_path),
                "index_file_path": str(self.index_file_path),
                "conversations_count": len(self.index.session_ids),
                "random_seed": self.index.random_seed,
                "sequential_iteration": self.index.use_sequential_iteration,
                "serialization_format": self.index.serialization_format.value,
                "total_size_bytes": self.index.total_size,
                "session_ids_sample": self.index.session_ids[:5]
                if self.index.session_ids
                else [],
            },
        )

    def __enter__(self) -> "MMapDatasetClient":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    @staticmethod
    def _cleanup_finalizer(
        data_mmap: mmap.mmap | None,
        index_mmap: mmap.mmap | None,
        data_file: Any | None,
        index_file: Any | None,
    ) -> None:
        """Resource cleanup method called during garbage collection."""
        resources = [data_mmap, index_mmap, data_file, index_file]
        for resource in resources:
            with suppress(Exception):
                if resource is not None:
                    resource.close()
                    _logger.debug("Finalizer cleaned up resource")

    def _cleanup_resources(self) -> None:
        """Clean up partially opened resources during error recovery."""
        for attr in ["data_mmap", "index_mmap", "data_file", "index_file"]:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if obj:
                    with suppress(Exception):
                        obj.close()

    def _load_conversations(self) -> dict[str, Conversation]:
        """Load and cache all conversations from memory-mapped file.

        Returns:
            Dictionary mapping session IDs to Conversation objects

        Raises:
            OSError: If reading from memory-mapped file fails
            MMapSerializationError: If conversation data is corrupted
        """
        if self._conversations_cache is None:
            try:
                # Read the serialized data from memory-mapped file
                self.data_mmap.seek(0)
                data_bytes = self.data_mmap.read(self.index.total_size)

                _logger.debug(
                    "Loading conversations from memory-mapped file",
                    extra={
                        "component": "MMapDatasetClient",
                        "operation": "load_conversations",
                        "data_size_bytes": len(data_bytes),
                        "serialization_format": self.index.serialization_format.value,
                    },
                )

                self._conversations_cache = (
                    MMapDatasetSerializer.deserialize_conversations(
                        data_bytes, self.index.serialization_format
                    )
                )

                _logger.debug(
                    "Conversations loaded and cached successfully",
                    extra={
                        "component": "MMapDatasetClient",
                        "conversations_loaded": len(self._conversations_cache),
                    },
                )

            except (OSError, MMapSerializationError) as e:
                _logger.error(
                    "Failed to load conversations from memory-mapped file",
                    extra={
                        "component": "MMapDatasetClient",
                        "error": str(e),
                        "data_file_path": str(self.data_file_path),
                    },
                )
                raise

        return self._conversations_cache

    def get_conversation(self, conversation_id: str | None = None) -> Conversation:
        """Get a conversation by ID or return a random/sequential one.

        Args:
            conversation_id: Optional specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If specific conversation_id is not found
            OSError: If reading from memory-mapped file fails
            ValueError: If conversation data is corrupted
        """
        conversations = self._load_conversations()

        if conversation_id is not None:
            if conversation_id not in conversations:
                raise KeyError(f"Conversation '{conversation_id}' not found in dataset")
            return conversations[conversation_id]

        # Return conversation based on iteration strategy
        if self.index.use_sequential_iteration:
            return self._get_sequential_conversation(conversations)
        else:
            return self._get_random_conversation(conversations)

    def _get_sequential_conversation(
        self, conversations: dict[str, Conversation]
    ) -> Conversation:
        """Get the next conversation in sequential order.

        Args:
            conversations: Dictionary of all conversations

        Returns:
            Next conversation in sequence (wraps around at end)
        """
        if self._sequential_iterator_index >= len(self.index.session_ids):
            _logger.debug(
                "Sequential iterator reached end, wrapping to beginning",
                extra={
                    "component": "MMapDatasetClient",
                    "operation": "sequential_iteration_wrap",
                    "total_conversations": len(self.index.session_ids),
                    "previous_index": self._sequential_iterator_index,
                },
            )
            self._sequential_iterator_index = 0

        session_id = self.index.session_ids[self._sequential_iterator_index]
        self._sequential_iterator_index += 1

        return conversations[session_id]

    def _get_random_conversation(
        self, conversations: dict[str, Conversation]
    ) -> Conversation:
        """Get a random conversation.

        Args:
            conversations: Dictionary of all conversations

        Returns:
            Randomly selected conversation
        """
        if self._conversation_query_random is not None:
            session_id = self._conversation_query_random.choice(self.index.session_ids)
        else:
            # Fallback to system random if no seed was provided
            session_id = random.choice(self.index.session_ids)

        return conversations[session_id]

    def close(self) -> None:
        """Close the memory-mapped files and associated resources.

        This method is safe to call multiple times.
        """
        resources = [
            (self.data_mmap, "data_mmap", MMapFileType.DATA),
            (self.index_mmap, "index_mmap", MMapFileType.INDEX),
            (self.data_file, "data_file", MMapFileType.DATA),
            (self.index_file, "index_file", MMapFileType.INDEX),
        ]

        for resource, attr_name, file_type in resources:
            if hasattr(self, attr_name) and resource:
                try:
                    resource.close()
                    _logger.debug(f"Closed {file_type.value} {attr_name}")
                except Exception as e:
                    _logger.warning(f"Error closing {file_type.value} {attr_name}: {e}")
                finally:
                    setattr(self, attr_name, None)

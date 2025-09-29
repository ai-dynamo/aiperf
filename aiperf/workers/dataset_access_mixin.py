# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.dataset.mmap_dataset_manager import MMapDatasetClient


class DatasetAccessMixin(AIPerfLoggerMixin):
    """
    Mixin that provides dataset access capabilities to worker processes.

    This mixin allows workers to access conversation data directly from memory-mapped files
    instead of making network requests to the dataset manager.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mixin with dataset access capabilities."""
        super().__init__(**kwargs)
        self._mmap_dataset_client: MMapDatasetClient | None = None
        self._mmap_enabled: bool = False

    def initialize_dataset_access(
        self, data_file_path: str | Path, index_file_path: str | Path
    ) -> None:
        """Initialize the memory-mapped dataset client with file paths.

        Args:
            data_file_path: Path to the memory-mapped data file
            index_file_path: Path to the memory-mapped index file

        Note:
            Logs warnings on failure but does not raise exceptions to allow
            graceful fallback to network-based dataset access.
        """
        try:
            self._mmap_dataset_client = MMapDatasetClient(
                str(data_file_path), str(index_file_path)
            )
            self._mmap_enabled = True
            self.debug(
                f"Initialized memory-mapped dataset client: "
                f"data_file={data_file_path}, index_file={index_file_path}"
            )
        except (FileNotFoundError, OSError) as e:
            self.warning(f"Failed to initialize memory-mapped dataset client: {e}")
            self._mmap_enabled = False
        except ValueError as e:
            self.warning(f"Invalid dataset files: {e}")
            self._mmap_enabled = False

    def get_conversation_from_dataset(
        self, conversation_id: str | None = None
    ) -> Conversation | None:
        """
        Get a conversation from memory-mapped files.

        Args:
            conversation_id: Optional conversation ID. If None, returns a random/sequential conversation.

        Returns:
            Conversation object if successful, None if memory mapping is not available or error occurs.
        """
        if not self._mmap_enabled or not self._mmap_dataset_client:
            self.debug(
                "Memory-mapped dataset not available, falling back to network access"
            )
            return None

        try:
            return self._mmap_dataset_client.get_conversation(conversation_id)
        except KeyError as e:
            self.warning(f"Conversation not found in dataset: {e}")
            return None
        except (OSError, ValueError) as e:
            self.warning(f"Error reading from memory-mapped dataset: {e}")
            return None

    def is_dataset_available(self) -> bool:
        """Check if memory-mapped dataset is available and ready for use.

        Returns:
            True if dataset is available, False otherwise
        """
        return self._mmap_enabled and self._mmap_dataset_client is not None

    def cleanup_dataset(self) -> None:
        """Clean up the memory-mapped dataset client and associated resources.

        This method is safe to call multiple times and will not raise exceptions.
        """
        if self._mmap_dataset_client:
            try:
                self._mmap_dataset_client.close()
                self.debug("Cleaned up memory-mapped dataset client")
            except Exception as e:
                self.warning(f"Error cleaning up memory-mapped dataset client: {e}")
            finally:
                self._mmap_dataset_client = None
                self._mmap_enabled = False

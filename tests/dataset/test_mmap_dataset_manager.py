# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from multiprocessing import Process

import pytest

from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import Text
from aiperf.dataset.mmap_dataset_manager import (
    MMapDatasetClient,
    MMapDatasetManager,
    MMapDatasetSerializer,
    MMapSerializationError,
    SerializationFormat,
)


class TestMMapDatasetSerializer:
    """Test the MMapDatasetSerializer utility functions."""

    def test_serialize_deserialize_conversations(self):
        """Test serialization and deserialization of conversations."""
        # Create test conversations
        conversations = {
            "session1": Conversation(
                session_id="session1",
                turns=[
                    Turn(timestamp=1000, texts=[Text(contents=["Hello"])]),
                    Turn(timestamp=2000, texts=[Text(contents=["World"])]),
                ],
            ),
            "session2": Conversation(
                session_id="session2",
                turns=[Turn(timestamp=3000, texts=[Text(contents=["Test"])])],
            ),
        }

        # Serialize
        serialized_data, format_type = MMapDatasetSerializer.serialize_conversations(
            conversations
        )
        assert isinstance(serialized_data, bytes)
        assert format_type in [
            SerializationFormat.JSON,
            SerializationFormat.JSON_COMPRESSED,
        ]

        # Deserialize
        deserialized_conversations = MMapDatasetSerializer.deserialize_conversations(
            serialized_data, format_type
        )

        # Verify
        assert len(deserialized_conversations) == 2
        assert "session1" in deserialized_conversations
        assert "session2" in deserialized_conversations
        assert deserialized_conversations["session1"].session_id == "session1"
        assert len(deserialized_conversations["session1"].turns) == 2
        assert deserialized_conversations["session2"].session_id == "session2"
        assert len(deserialized_conversations["session2"].turns) == 1

    def test_create_index(self):
        """Test creation of dataset index."""
        conversations = {
            "session1": Conversation(session_id="session1", turns=[Turn()]),
            "session2": Conversation(session_id="session2", turns=[Turn()]),
        }

        serialized_data, format_type = MMapDatasetSerializer.serialize_conversations(
            conversations
        )
        index = MMapDatasetSerializer.create_index(
            conversations, serialized_data, format_type
        )

        assert len(index.session_ids) == 2
        assert "session1" in index.session_ids
        assert "session2" in index.session_ids
        assert index.total_size == len(serialized_data)

    def test_serialize_empty_conversations(self):
        """Test serialization with empty conversations dict."""
        with pytest.raises(
            MMapSerializationError,
            match="Cannot serialize empty conversations dictionary",
        ):
            MMapDatasetSerializer.serialize_conversations({})

    def test_create_index_empty_conversations(self):
        """Test index creation with empty conversations."""
        with pytest.raises(
            MMapSerializationError, match="Cannot create index for empty conversations"
        ):
            MMapDatasetSerializer.create_index({}, b"some_data")

    def test_create_index_empty_data(self):
        """Test index creation with empty serialized data."""
        conversations = {
            "session1": Conversation(session_id="session1", turns=[Turn()])
        }
        with pytest.raises(
            MMapSerializationError,
            match="Cannot create index for empty serialized data",
        ):
            MMapDatasetSerializer.create_index(conversations, b"")

    def test_serialize_with_compression(self):
        """Test serialization with compression enabled."""
        conversations = {
            "session1": Conversation(
                session_id="session1",
                turns=[Turn(timestamp=1000, texts=[Text(contents=["Hello World"])])],
            ),
        }

        # Test without compression
        uncompressed_data, uncompressed_format = (
            MMapDatasetSerializer.serialize_conversations(
                conversations, compression=False
            )
        )
        assert uncompressed_format == SerializationFormat.JSON

        # Test with compression
        compressed_data, compressed_format = (
            MMapDatasetSerializer.serialize_conversations(
                conversations, compression=True
            )
        )
        assert compressed_format == SerializationFormat.JSON_COMPRESSED

        # Compressed data should be smaller for larger datasets
        # For small datasets, compression might not reduce size due to overhead
        assert isinstance(compressed_data, bytes)
        assert isinstance(uncompressed_data, bytes)

        # Both should deserialize to the same result
        uncompressed_result = MMapDatasetSerializer.deserialize_conversations(
            uncompressed_data, uncompressed_format
        )
        compressed_result = MMapDatasetSerializer.deserialize_conversations(
            compressed_data, compressed_format
        )

        assert uncompressed_result == compressed_result


class TestMMapDatasetManager:
    """Test the MMapDatasetManager class."""

    @pytest.fixture
    def sample_conversations(self):
        """Create sample conversations for testing."""
        return {
            "session1": Conversation(
                session_id="session1",
                turns=[Turn(timestamp=1000, texts=[Text(contents=["Hello"])])],
            ),
            "session2": Conversation(
                session_id="session2",
                turns=[Turn(timestamp=2000, texts=[Text(contents=["World"])])],
            ),
        }

    def test_initialization(self, sample_conversations):
        """Test MMapDatasetManager initialization."""
        manager = MMapDatasetManager(
            dataset=sample_conversations, random_seed=42, use_sequential_iteration=True
        )

        assert manager.dataset == sample_conversations
        assert manager.random_seed == 42
        assert manager.index.use_sequential_iteration is True
        assert manager.index.random_seed == 42
        assert len(manager.index.session_ids) == 2

    def test_create_memory_mapped_files(self, sample_conversations):
        """Test creation of memory-mapped files."""
        with MMapDatasetManager(dataset=sample_conversations) as manager:
            data_file_path, index_file_path = manager.create_memory_mapped_files()

            assert isinstance(data_file_path, str)
            assert isinstance(index_file_path, str)
            assert manager.data_file_path is not None
            assert manager.index_file_path is not None
            assert manager.data_file_path.exists()
            assert manager.index_file_path.exists()

    def test_cleanup_memory_mapped_files(self, sample_conversations):
        """Test cleanup of memory-mapped files."""
        manager = MMapDatasetManager(dataset=sample_conversations)

        # Create and then cleanup
        manager.create_memory_mapped_files()
        manager.cleanup_memory_mapped_files()

        assert manager.data_file_path is None
        assert manager.index_file_path is None

    def test_context_manager(self, sample_conversations):
        """Test context manager functionality."""
        data_file_path = None
        index_file_path = None

        with MMapDatasetManager(dataset=sample_conversations) as manager:
            data_file_path, index_file_path = manager.create_memory_mapped_files()
            assert manager.data_file_path.exists()
            assert manager.index_file_path.exists()

        # Files should be cleaned up after context manager exit
        from pathlib import Path

        assert not Path(data_file_path).exists()
        assert not Path(index_file_path).exists()


class TestMMapDatasetClient:
    """Test the MMapDatasetClient class."""

    @pytest.fixture
    def mmap_setup(self):
        """Setup memory-mapped files for testing."""
        conversations = {
            "session1": Conversation(
                session_id="session1",
                turns=[Turn(timestamp=1000, texts=[Text(contents=["Hello"])])],
            ),
            "session2": Conversation(
                session_id="session2",
                turns=[Turn(timestamp=2000, texts=[Text(contents=["World"])])],
            ),
        }

        manager = MMapDatasetManager(
            dataset=conversations, random_seed=42, use_sequential_iteration=False
        )

        data_file_path, index_file_path = manager.create_memory_mapped_files()

        yield data_file_path, index_file_path, conversations

        manager.cleanup_memory_mapped_files()

    def test_initialization(self, mmap_setup):
        """Test MMapDatasetClient initialization."""
        data_file_path, index_file_path, _ = mmap_setup

        with MMapDatasetClient(data_file_path, index_file_path) as client:
            assert str(client.data_file_path) == data_file_path
            assert str(client.index_file_path) == index_file_path
            assert len(client.index.session_ids) == 2
            assert client.index.random_seed == 42
            assert client.index.use_sequential_iteration is False

    def test_get_conversation_by_id(self, mmap_setup):
        """Test getting a specific conversation by ID."""
        data_file_path, index_file_path, original_conversations = mmap_setup

        with MMapDatasetClient(data_file_path, index_file_path) as client:
            conversation = client.get_conversation("session1")

            assert conversation.session_id == "session1"
            assert len(conversation.turns) == 1
            assert conversation.turns[0].timestamp == 1000

    def test_get_conversation_by_id_not_found(self, mmap_setup):
        """Test getting a conversation with non-existent ID."""
        data_file_path, index_file_path, _ = mmap_setup

        with (
            MMapDatasetClient(data_file_path, index_file_path) as client,
            pytest.raises(KeyError, match="'non_existent' not found in dataset"),
        ):
            client.get_conversation("non_existent")

    def test_get_random_conversation(self, mmap_setup):
        """Test getting a random conversation."""
        data_file_path, index_file_path, _ = mmap_setup

        with MMapDatasetClient(data_file_path, index_file_path) as client:
            conversation = client.get_conversation()

            assert conversation.session_id in ["session1", "session2"]
            assert len(conversation.turns) >= 1

    def test_sequential_iteration(self):
        """Test sequential conversation iteration."""
        conversations = {
            "session1": Conversation(session_id="session1", turns=[Turn()]),
            "session2": Conversation(session_id="session2", turns=[Turn()]),
            "session3": Conversation(session_id="session3", turns=[Turn()]),
        }

        with MMapDatasetManager(
            dataset=conversations, use_sequential_iteration=True
        ) as manager:
            data_file_path, index_file_path = manager.create_memory_mapped_files()

            with MMapDatasetClient(data_file_path, index_file_path) as client:
                # Get conversations sequentially
                conv1 = client.get_conversation()
                conv2 = client.get_conversation()
                conv3 = client.get_conversation()
                conv4 = client.get_conversation()  # Should wrap around

                # Verify sequential order (should be same as session_ids order)
                session_ids = client.index.session_ids
                assert conv1.session_id == session_ids[0]
                assert conv2.session_id == session_ids[1]
                assert conv3.session_id == session_ids[2]
                assert conv4.session_id == session_ids[0]  # Wrapped around


def worker_process_test(data_file_path: str, index_file_path: str, results_queue):
    """Worker process function for multiprocess testing."""
    try:
        with MMapDatasetClient(data_file_path, index_file_path) as client:
            # Get a conversation
            conversation = client.get_conversation()
            results_queue.put(
                {
                    "success": True,
                    "session_id": conversation.session_id,
                    "turns_count": len(conversation.turns),
                }
            )

    except Exception as e:
        results_queue.put({"success": False, "error": str(e)})


class TestMultiprocessAccess:
    """Test multiprocess access to memory-mapped dataset."""

    def test_multiple_workers_access(self):
        """Test that multiple worker processes can access the memory-mapped dataset."""
        from multiprocessing import Queue

        conversations = {
            "session1": Conversation(session_id="session1", turns=[Turn()]),
            "session2": Conversation(session_id="session2", turns=[Turn()]),
        }

        with MMapDatasetManager(dataset=conversations, random_seed=42) as manager:
            data_file_path, index_file_path = manager.create_memory_mapped_files()

            # Create multiple worker processes
            num_workers = 3
            processes = []
            results_queue = Queue()

            for _ in range(num_workers):
                p = Process(
                    target=worker_process_test,
                    args=(data_file_path, index_file_path, results_queue),
                )
                p.start()
                processes.append(p)

            # Wait for all processes and collect results
            for p in processes:
                p.join()

            results = []
            while not results_queue.empty():
                results.append(results_queue.get())

            # Verify all workers succeeded
            assert len(results) == num_workers
            for result in results:
                assert result["success"] is True
                assert result["session_id"] in ["session1", "session2"]
                assert result["turns_count"] == 1

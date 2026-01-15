# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DatasetMetadata and DatasetConfiguredNotification flow."""

from aiperf.common.models import ConversationMetadata, TurnMetadata
from tests.unit.timing.conftest import (
    create_mock_dataset_metadata,
    create_mock_dataset_metadata_with_schedule,
)


class TestDatasetMetadataIntegration:
    """Integration tests for dataset metadata usage."""

    def test_metadata_from_schedule_creates_multi_turn_conversations(self):
        """Test creating dataset metadata from a schedule with multi-turn conversations."""
        schedule = [
            (0, "conv1"),
            (100, "conv2"),
            (150, "conv1"),  # Second turn for conv1
            (200, "conv1"),  # Third turn for conv1
            (250, "conv2"),  # Second turn for conv2
        ]

        metadata = create_mock_dataset_metadata_with_schedule(schedule)
        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert len(metadata.conversations) == 2
        assert len(conv_dict["conv1"].turns) == 3
        assert conv_dict["conv1"].turns[0].timestamp_ms == 0
        assert [turn.delay_ms for turn in conv_dict["conv1"].turns[1:]] == [150, 50]
        assert len(conv_dict["conv2"].turns) == 2

    def test_metadata_extraction_for_fixed_schedule(self):
        """Test extracting timing information from metadata for fixed schedule strategy."""
        schedule = [(0, "conv1"), (100, "conv2"), (200, "conv3")]
        metadata = create_mock_dataset_metadata_with_schedule(schedule)

        # Simulate what FixedScheduleStrategy does
        extracted_schedule = []
        for conv in metadata.conversations:
            if conv.turns and conv.turns[0].timestamp_ms is not None:
                extracted_schedule.append(
                    (conv.turns[0].timestamp_ms, conv.conversation_id)
                )
        extracted_schedule.sort(key=lambda x: x[0])

        assert extracted_schedule == schedule

    def test_metadata_with_mixed_turn_counts(self):
        """Test metadata with conversations having different turn counts."""
        metadata = create_mock_dataset_metadata(
            conversation_ids=["single", "double", "triple"],
            turn_counts=[1, 2, 3],
        )
        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert len(conv_dict["single"].turns) == 1
        assert len(conv_dict["double"].turns) == 2
        assert len(conv_dict["triple"].turns) == 3


class TestFloatingPointTimestampPreservation:
    """Tests to ensure floating point timestamps are preserved."""

    def test_conversation_metadata_preserves_float_timestamps(self):
        """Test that ConversationMetadata preserves floating point timestamps in turns."""
        turns = [
            TurnMetadata(timestamp_ms=0.0, delay_ms=None),
            TurnMetadata(timestamp_ms=100.5, delay_ms=100.5),
            TurnMetadata(timestamp_ms=200.75, delay_ms=100.25),
        ]
        conv = ConversationMetadata(conversation_id="test-conv", turns=turns)

        assert conv.turns[0].timestamp_ms == 0.0
        assert conv.turns[1].timestamp_ms == 100.5
        assert conv.turns[1].delay_ms == 100.5
        assert conv.turns[2].timestamp_ms == 200.75
        assert conv.turns[2].delay_ms == 100.25

    def test_dataset_metadata_preserves_float_timestamps(self):
        """Test that DatasetMetadata preserves floating point timestamps across conversations."""
        metadata = create_mock_dataset_metadata_with_schedule(
            [
                (0.0, "conv1"),
                (100.5, "conv2"),
                (150.75, "conv1"),  # Second turn
            ]
        )
        conv_dict = {conv.conversation_id: conv for conv in metadata.conversations}

        assert conv_dict["conv1"].turns[0].timestamp_ms == 0.0
        assert conv_dict["conv1"].turns[1].timestamp_ms == 150.75
        assert conv_dict["conv2"].turns[0].timestamp_ms == 100.5

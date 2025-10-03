# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid

from aiperf.common.enums import CreditPhase
from aiperf.common.models import RequestRecord


class TestRequestRecordXRequestID:
    """Test RequestRecord x_request_id functionality."""

    def test_x_request_id_optional(self):
        """Test that x_request_id is optional and defaults to None."""
        record = RequestRecord()

        assert record.x_request_id is None

    def test_x_request_id_can_be_set(self):
        """Test that x_request_id can be set."""
        request_id = str(uuid.uuid4())
        record = RequestRecord(x_request_id=request_id)

        assert record.x_request_id == request_id

    def test_x_request_id_serialization(self):
        """Test that x_request_id is properly serialized."""
        request_id = str(uuid.uuid4())
        record = RequestRecord(x_request_id=request_id)

        serialized = record.model_dump()
        assert serialized["x_request_id"] == request_id

        json_str = record.model_dump_json()
        assert request_id in json_str

    def test_x_request_id_deserialization(self):
        """Test that x_request_id is properly deserialized."""
        request_id = str(uuid.uuid4())
        original_record = RequestRecord(x_request_id=request_id)

        json_str = original_record.model_dump_json()
        restored_record = RequestRecord.model_validate_json(json_str)

        assert restored_record.x_request_id == request_id

    def test_x_request_id_with_other_fields(self):
        """Test x_request_id works with other RequestRecord fields."""
        request_id = str(uuid.uuid4())
        timestamp = time.time_ns()
        record = RequestRecord(
            x_request_id=request_id,
            timestamp_ns=timestamp,
            model_name="test-model",
            conversation_id="conv-123",
            turn_index=0,
            credit_phase=CreditPhase.PROFILING,
        )

        assert record.x_request_id == request_id
        assert record.timestamp_ns == timestamp
        assert record.model_name == "test-model"
        assert record.conversation_id == "conv-123"


class TestRequestRecordXCorrelationID:
    """Test RequestRecord x_correlation_id functionality."""

    def test_x_correlation_id_optional(self):
        """Test that x_correlation_id is optional and defaults to None."""
        record = RequestRecord()

        assert record.x_correlation_id is None

    def test_x_correlation_id_can_be_set(self):
        """Test that x_correlation_id can be set."""
        correlation_id = str(uuid.uuid4())
        record = RequestRecord(x_correlation_id=correlation_id)

        assert record.x_correlation_id == correlation_id

    def test_x_correlation_id_serialization(self):
        """Test that x_correlation_id is properly serialized."""
        correlation_id = str(uuid.uuid4())
        record = RequestRecord(x_correlation_id=correlation_id)

        serialized = record.model_dump()
        assert serialized["x_correlation_id"] == correlation_id

        json_str = record.model_dump_json()
        assert correlation_id in json_str

    def test_x_correlation_id_deserialization(self):
        """Test that x_correlation_id is properly deserialized."""
        correlation_id = str(uuid.uuid4())
        original_record = RequestRecord(x_correlation_id=correlation_id)

        json_str = original_record.model_dump_json()
        restored_record = RequestRecord.model_validate_json(json_str)

        assert restored_record.x_correlation_id == correlation_id

    def test_x_correlation_id_with_other_fields(self):
        """Test x_correlation_id works with other RequestRecord fields."""
        correlation_id = str(uuid.uuid4())
        timestamp = time.time_ns()
        record = RequestRecord(
            x_correlation_id=correlation_id,
            timestamp_ns=timestamp,
            model_name="test-model",
            conversation_id="conv-123",
            turn_index=0,
            credit_phase=CreditPhase.PROFILING,
        )

        assert record.x_correlation_id == correlation_id
        assert record.timestamp_ns == timestamp
        assert record.model_name == "test-model"
        assert record.conversation_id == "conv-123"


class TestRequestRecordBothIDs:
    """Test RequestRecord with both x_request_id and x_correlation_id."""

    def test_both_ids_can_be_set(self):
        """Test that both x_request_id and x_correlation_id can be set simultaneously."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        record = RequestRecord(
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        assert record.x_request_id == request_id
        assert record.x_correlation_id == correlation_id

    def test_both_ids_are_independent(self):
        """Test that x_request_id and x_correlation_id are independent."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        record = RequestRecord(
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        assert record.x_request_id != record.x_correlation_id

    def test_both_ids_serialization(self):
        """Test that both IDs are properly serialized."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        record = RequestRecord(
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        serialized = record.model_dump()
        assert serialized["x_request_id"] == request_id
        assert serialized["x_correlation_id"] == correlation_id

    def test_both_ids_deserialization(self):
        """Test that both IDs are properly deserialized."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        original_record = RequestRecord(
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        json_str = original_record.model_dump_json()
        restored_record = RequestRecord.model_validate_json(json_str)

        assert restored_record.x_request_id == request_id
        assert restored_record.x_correlation_id == correlation_id

    def test_typical_workflow(self):
        """Test typical workflow with both IDs matching credit drop/return pattern."""
        drop_request_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        record = RequestRecord(
            x_request_id=request_id,
            x_correlation_id=drop_request_id,
            timestamp_ns=time.time_ns(),
            start_perf_ns=time.perf_counter_ns(),
            model_name="test-model",
            conversation_id="conv-123",
            turn_index=0,
            credit_phase=CreditPhase.PROFILING,
        )

        assert record.x_request_id == request_id
        assert record.x_correlation_id == drop_request_id
        assert record.x_request_id != record.x_correlation_id

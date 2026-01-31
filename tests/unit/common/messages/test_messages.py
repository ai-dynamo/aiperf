# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import orjson
import pytest

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.messages import (
    ErrorMessage,
    HeartbeatMessage,
    ShutdownCommand,
    StatusMessage,
)
from aiperf.common.models import ErrorDetails
from aiperf.plugin.enums import ServiceType


def test_status_message():
    message = StatusMessage(
        state=LifecycleState.RUNNING,
        service_id="test",
        service_type=ServiceType.WORKER,
        request_ns=1234567890,
        request_id="test",
    )
    assert message.model_dump(exclude_none=True) == {
        "message_type": MessageType.STATUS,
        "state": LifecycleState.RUNNING,
        "service_id": "test",
        "service_type": ServiceType.WORKER,
        "request_ns": 1234567890,
        "request_id": "test",
    }
    assert json.loads(message.model_dump_json(exclude_none=True)) == json.loads(
        '{"message_type":"status","state":"running","service_id":"test","service_type":"worker","request_ns":1234567890,"request_id":"test"}'
    )

    message = StatusMessage(
        state=LifecycleState.INITIALIZED,
        request_ns=1234567890,
        request_id=None,
        service_id="test",
        service_type=ServiceType.WORKER,
    )
    assert message.model_dump(exclude_none=True) == {
        "message_type": MessageType.STATUS,
        "state": LifecycleState.INITIALIZED,
        "service_id": "test",
        "service_type": ServiceType.WORKER,
        "request_ns": 1234567890,
    }
    assert json.loads(message.model_dump_json(exclude_none=True)) == json.loads(
        '{"message_type":"status","state":"initialized","service_id":"test","service_type":"worker","request_ns":1234567890}'
    )


class TestMessageToJsonBytes:
    """Test suite for Message.to_json_bytes() optimization."""

    def test_to_json_bytes_returns_bytes(self):
        """Test that to_json_bytes() returns bytes type."""
        message = ShutdownCommand(
            service_id="test-service",
            request_id="test-request",
        )
        result = message.to_json_bytes()
        assert isinstance(result, bytes)

    def test_to_json_bytes_excludes_none_fields(self):
        """Test that to_json_bytes() automatically excludes None fields."""
        # Create message with some None fields
        message = StatusMessage(
            state=LifecycleState.RUNNING,
            service_id="test",
            service_type=ServiceType.WORKER,
            request_ns=1234567890,
            request_id=None,  # This should be excluded
        )

        json_bytes = message.to_json_bytes()
        parsed = orjson.loads(json_bytes)

        # request_id should not be in the output
        assert "request_id" not in parsed
        assert "message_type" in parsed
        assert "state" in parsed

    def test_to_json_bytes_includes_non_none_fields(self):
        """Test that to_json_bytes() includes all non-None fields."""
        message = StatusMessage(
            state=LifecycleState.INITIALIZED,
            service_id="test-service",
            service_type=ServiceType.WORKER_MANAGER,
            request_ns=9876543210,
            request_id="req-123",
        )

        json_bytes = message.to_json_bytes()
        parsed = orjson.loads(json_bytes)

        assert parsed["message_type"] == "status"
        assert parsed["state"] == "initialized"
        assert parsed["service_id"] == "test-service"
        assert parsed["service_type"] == "worker_manager"
        assert parsed["request_ns"] == 9876543210
        assert parsed["request_id"] == "req-123"

    def test_to_json_bytes_roundtrip_with_from_json(self):
        """Test that to_json_bytes() output can be deserialized with from_json()."""
        original = HeartbeatMessage(
            service_id="worker-1",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
            request_id="heartbeat-001",
        )

        # Serialize and deserialize
        json_bytes = original.to_json_bytes()
        restored = HeartbeatMessage.from_json(json_bytes)

        # Check all fields match
        assert restored.message_type == original.message_type
        assert restored.service_id == original.service_id
        assert restored.service_type == original.service_type
        assert restored.state == original.state
        assert restored.request_id == original.request_id

    def test_to_json_bytes_equivalent_to_model_dump_json(self):
        """Test that to_json_bytes() produces equivalent output to model_dump_json(exclude_none=True)."""
        message = ShutdownCommand(
            service_id="controller",
            request_id="shutdown-001",
        )

        # Old way
        old_bytes = message.model_dump_json(exclude_none=True).encode("utf-8")
        old_parsed = json.loads(old_bytes)

        # New way
        new_bytes = message.to_json_bytes()
        new_parsed = orjson.loads(new_bytes)

        # Should produce equivalent JSON
        assert old_parsed == new_parsed

    def test_to_json_bytes_with_complex_nested_data(self):
        """Test to_json_bytes() with complex nested structures."""
        error_details = ErrorDetails(
            type="TestError",
            message="This is a test error with complex data",
            code=500,
            details={
                "nested": {
                    "data": ["item1", "item2", "item3"],
                    "count": 42,
                },
                "metadata": {"key": "value"},
            },
        )

        message = ErrorMessage(
            request_id="error-001",
            error=error_details,
        )

        json_bytes = message.to_json_bytes()
        parsed = orjson.loads(json_bytes)

        # Verify nested structure is preserved
        assert parsed["error"]["type"] == "TestError"
        assert parsed["error"]["code"] == 500
        assert parsed["error"]["details"]["nested"]["data"] == [
            "item1",
            "item2",
            "item3",
        ]
        assert parsed["error"]["details"]["nested"]["count"] == 42
        assert parsed["error"]["details"]["metadata"]["key"] == "value"

    def test_to_json_bytes_with_large_message(self):
        """Test to_json_bytes() with a large message (tests performance scenario)."""
        # Create a large error message similar to benchmark
        large_details = {f"metric_{i}": f"value_{i}" * 10 for i in range(100)}

        message = ErrorMessage(
            request_id="large-error-001",
            error=ErrorDetails(
                type="LargeError",
                message="Large error message " * 50,
                code=1000,
                details=large_details,
            ),
        )

        json_bytes = message.to_json_bytes()

        # Verify it's substantial
        assert len(json_bytes) > 5000  # Should be multiple KB

        # Verify it can be deserialized
        restored = ErrorMessage.from_json(json_bytes)
        assert restored.request_id == "large-error-001"
        assert restored.error.type == "LargeError"
        assert len(restored.error.details) == 100

    def test_to_json_bytes_multiple_messages_independence(self):
        """Test that to_json_bytes() calls don't interfere with each other."""
        msg1 = ShutdownCommand(service_id="service-1", request_id="req-1")
        msg2 = HeartbeatMessage(
            service_id="service-2",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
            request_id="req-2",
        )

        bytes1 = msg1.to_json_bytes()
        bytes2 = msg2.to_json_bytes()

        # They should be different
        assert bytes1 != bytes2

        # Each should deserialize to correct type
        restored1 = ShutdownCommand.from_json(bytes1)
        restored2 = HeartbeatMessage.from_json(bytes2)

        assert restored1.service_id == "service-1"
        assert restored2.service_id == "service-2"

    def test_to_json_bytes_uses_orjson(self):
        """Test that to_json_bytes() output is valid orjson format."""
        message = StatusMessage(
            state=LifecycleState.RUNNING,
            service_id="test",
            service_type=ServiceType.WORKER,
            request_ns=1234567890,
        )

        json_bytes = message.to_json_bytes()

        # Should be parseable by orjson
        parsed = orjson.loads(json_bytes)
        assert isinstance(parsed, dict)
        assert "message_type" in parsed

    def test_to_json_bytes_empty_optional_fields(self):
        """Test to_json_bytes() with minimal required fields only."""
        message = ShutdownCommand(
            service_id="minimal",
            # request_id omitted (None by default)
        )

        json_bytes = message.to_json_bytes()
        parsed = orjson.loads(json_bytes)

        # Should only contain required fields and message_type
        assert "service_id" in parsed
        assert "message_type" in parsed
        assert "request_id" not in parsed  # Should be excluded due to exclude_none

    @pytest.mark.parametrize(
        "message_type,kwargs",
        [
            (ShutdownCommand, {"service_id": "test"}),
            (
                StatusMessage,
                {
                    "service_id": "test",
                    "service_type": ServiceType.WORKER,
                    "state": LifecycleState.RUNNING,
                },
            ),
            (
                HeartbeatMessage,
                {
                    "service_id": "test",
                    "service_type": ServiceType.SYSTEM_CONTROLLER,
                    "state": LifecycleState.INITIALIZED,
                },
            ),
        ],
    )  # fmt: skip
    def test_to_json_bytes_various_message_types(self, message_type, kwargs):
        """Test to_json_bytes() works with various message types."""
        message = message_type(**kwargs)
        json_bytes = message.to_json_bytes()

        # Should produce valid bytes
        assert isinstance(json_bytes, bytes)
        assert len(json_bytes) > 0

        # Should be deserializable
        restored = message_type.from_json(json_bytes)
        assert restored.message_type == message.message_type


class TestMessageStringRepresentation:
    """Test suite for Message.__str__() method (uses model_dump_json with exclude_none)."""

    @pytest.mark.parametrize(
        "message,expected_present,expected_absent",
        [
            # Test None field exclusion
            (
                StatusMessage(
                    state=LifecycleState.RUNNING,
                    service_id="test",
                    service_type=ServiceType.WORKER,
                    request_ns=1234567890,
                    request_id=None,
                ),
                {"message_type", "state", "service_id"},
                {"request_id"},
            ),
            # Test all fields present
            (
                HeartbeatMessage(
                    service_id="worker-1",
                    service_type=ServiceType.WORKER,
                    state=LifecycleState.RUNNING,
                    request_id="heartbeat-001",
                    request_ns=9876543210,
                ),
                {"message_type", "service_id", "request_id", "request_ns"},
                set(),
            ),
            # Test with complex nested structures
            (
                ErrorMessage(
                    request_id="error-123",
                    error=ErrorDetails(
                        type="ComplexError",
                        message="Complex error message",
                        code=500,
                        details={"nested": {"data": [1, 2, 3]}},
                    ),
                ),
                {"message_type", "error"},
                set(),
            ),
        ],
    )  # fmt: skip
    def test_message_str_json_output(self, message, expected_present, expected_absent):
        """Test that __str__() returns valid JSON with correct field inclusion/exclusion."""
        str_output = str(message)
        parsed = json.loads(str_output)

        # Check expected fields are present
        for field in expected_present:
            assert field in parsed, f"Expected field '{field}' not in output"

        # Check expected fields are absent
        for field in expected_absent:
            assert field not in parsed, f"Unexpected field '{field}' in output"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import time
from dataclasses import fields, is_dataclass

import msgspec
import orjson
import pytest

from aiperf.common.enums import (
    CommAddress,
    ConversationContextMode,
    LifecycleState,
    MessageType,
)
from aiperf.common.messages import (
    ErrorMessage,
    HeartbeatMessage,
    StatusMessage,
)
from aiperf.common.models import ErrorDetails
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateQuery,
    GroupDatasetStateSnapshot,
    GroupManagerToPeerMessage,
    GroupPeerAck,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupPeerHello,
    GroupPeerShutdown,
    GroupWorkerHealth,
    GroupWorkerStartupState,
    PeerToGroupManagerMessage,
)
from aiperf.plugin.enums import ServiceType
from aiperf.workers.group_dataset_authority import GroupDatasetSnapshot


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


class TestBaseStatusMessageTimestamp:
    """Tests for BaseStatusMessage default_factory timestamp behavior."""

    def test_request_ns_differs_between_instances(self):
        """Each instance gets its own timestamp via default_factory, not a shared class-level value."""
        msg1 = StatusMessage(
            state=LifecycleState.RUNNING,
            service_id="svc-1",
            service_type=ServiceType.WORKER,
        )
        msg2 = StatusMessage(
            state=LifecycleState.RUNNING,
            service_id="svc-2",
            service_type=ServiceType.WORKER,
        )
        assert msg1.request_ns != msg2.request_ns, (
            "default_factory should produce unique timestamps per instance"
        )

    def test_request_ns_is_recent(self):
        """Auto-filled timestamp is close to current time."""
        before = time.time_ns()
        msg = HeartbeatMessage(
            state=LifecycleState.RUNNING,
            service_id="svc",
            service_type=ServiceType.WORKER,
        )
        after = time.time_ns()
        assert before <= msg.request_ns <= after

    def test_explicit_request_ns_not_overwritten(self):
        """Explicitly provided request_ns is preserved."""
        msg = StatusMessage(
            state=LifecycleState.RUNNING,
            service_id="svc",
            service_type=ServiceType.WORKER,
            request_ns=42,
        )
        assert msg.request_ns == 42


class TestMessageToJsonBytes:
    """Test suite for Message.to_json_bytes() optimization."""

    def test_to_json_bytes_returns_bytes(self):
        """Test that to_json_bytes() returns bytes type."""
        message = HeartbeatMessage(
            service_id="test-service",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
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
        message = HeartbeatMessage(
            service_id="controller",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
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
        msg1 = HeartbeatMessage(
            service_id="service-1",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
            request_id="req-1",
        )
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
        restored1 = HeartbeatMessage.from_json(bytes1)
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
        message = HeartbeatMessage(
            service_id="minimal",
            service_type=ServiceType.WORKER,
            state=LifecycleState.RUNNING,
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
            (HeartbeatMessage, {"service_id": "test", "service_type": ServiceType.WORKER, "state": LifecycleState.RUNNING}),
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


class TestGroupLifecycleWireContract:
    """Focused contract tests for group-local lifecycle msgspec wire models."""

    def test_group_dataset_snapshot_is_local_only_dataclass(self) -> None:
        """The in-memory dataset snapshot should stay on the local-only model side."""
        snapshot = GroupDatasetSnapshot(
            benchmark_generation="bench-1",
            dataset_generation="dataset-1",
            ready=True,
            error_message="none",
        )

        assert is_dataclass(snapshot)
        assert GroupDatasetSnapshot.__slots__ == (
            "benchmark_generation",
            "dataset_generation",
            "ready",
            "error_message",
        )
        assert [field.name for field in fields(snapshot)] == [
            "benchmark_generation",
            "dataset_generation",
            "ready",
            "error_message",
        ]
        assert not hasattr(GroupDatasetSnapshot, "__struct_fields__")
        assert snapshot.ready is True

    @pytest.mark.parametrize(
        ("message", "union_type"),
        [
            pytest.param(
                GroupPeerAck(service_id="worker-0"),
                GroupManagerToPeerMessage,
                id="ack",
            ),
            pytest.param(
                GroupDatasetReady(
                    service_id="worker-group-manager-0",
                    data_file_path="/tmp/data.bin",
                    index_file_path="/tmp/index.bin",
                    conversation_count=4,
                    total_size_bytes=128,
                ),
                GroupManagerToPeerMessage,
                id="dataset-ready",
            ),
            pytest.param(
                GroupDatasetStateSnapshot(
                    rid="rid-1",
                    service_id="worker-group-manager-0",
                    benchmark_generation="bench-1",
                    dataset_generation="dataset-1",
                    default_context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
                    data_file_path="/tmp/data.bin",
                    index_file_path="/tmp/index.bin",
                    conversation_count=12,
                    total_size_bytes=345,
                    ready=True,
                ),
                GroupManagerToPeerMessage,
                id="dataset-snapshot",
            ),
            pytest.param(
                GroupPeerCommand(
                    cid="cmd-1",
                    service_id="worker-0",
                    command="shutdown",
                ),
                GroupManagerToPeerMessage,
                id="command",
            ),
            pytest.param(
                GroupPeerHello(
                    service_id="worker-0",
                    service_type="worker",
                    pod_index="0",
                ),
                PeerToGroupManagerMessage,
                id="hello",
            ),
            pytest.param(
                GroupPeerShutdown(
                    service_id="worker-0",
                    service_type="worker",
                ),
                PeerToGroupManagerMessage,
                id="shutdown",
            ),
            pytest.param(
                GroupWorkerHealth(
                    service_id="worker-0",
                    pid=123,
                    create_time=1.0,
                    uptime=2.0,
                    cpu_usage=3.5,
                    memory_usage=4096,
                    pss_memory=2048,
                    io_counters=(1, 2, 3, 4, 5, 6),
                    cpu_times=(0.1, 0.2, 0.3),
                    num_ctx_switches=(7, 8),
                    num_threads=9,
                    task_total=10,
                    task_failed=1,
                    task_completed=8,
                ),
                PeerToGroupManagerMessage,
                id="worker-health",
            ),
            pytest.param(
                GroupWorkerHealth(
                    service_id="worker-1",
                    create_time=2.0,
                    uptime=5.0,
                    cpu_usage=10.0,
                    memory_usage=8192,
                    task_total=0,
                    task_failed=0,
                    task_completed=0,
                ),
                PeerToGroupManagerMessage,
                id="worker-health-minimal",
            ),
            pytest.param(
                GroupDatasetStateSnapshot(
                    rid="rid-2",
                    service_id="worker-group-manager-1",
                ),
                GroupManagerToPeerMessage,
                id="dataset-snapshot-defaults",
            ),
            pytest.param(
                GroupDatasetReady(
                    service_id="worker-group-manager-0",
                    data_file_path="/tmp/data.bin",
                    index_file_path="/tmp/index.bin",
                    conversation_count=4,
                    total_size_bytes=128,
                    success=False,
                    error_message="download failed",
                ),
                GroupManagerToPeerMessage,
                id="dataset-ready-failure",
            ),
            pytest.param(
                GroupWorkerStartupState(
                    service_id="worker-0",
                    startup_state="ready",
                    request_ns=123,
                ),
                PeerToGroupManagerMessage,
                id="worker-startup-state",
            ),
            pytest.param(
                GroupDatasetStateQuery(
                    rid="rid-1",
                    service_id="worker-0",
                ),
                PeerToGroupManagerMessage,
                id="dataset-query",
            ),
            pytest.param(
                GroupPeerCommandAck(
                    cid="cmd-1",
                    service_id="worker-0",
                ),
                PeerToGroupManagerMessage,
                id="command-ack",
            ),
        ],
    )  # fmt: skip
    def test_group_lifecycle_tagged_unions_decode_all_wire_variants(
        self,
        message: object,
        union_type: object,
    ) -> None:
        """Every wire struct should decode through its canonical tagged union."""
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(union_type)

        restored = decoder.decode(encoder.encode(message))

        assert hasattr(type(message), "__struct_fields__")
        assert restored == message
        assert isinstance(restored, type(message))

    def test_group_dataset_state_snapshot_maps_cleanly_to_local_snapshot(self) -> None:
        """Wire snapshots should project onto the local-only dataset snapshot fields."""
        source_snapshot = GroupDatasetSnapshot(
            benchmark_generation="bench-1",
            dataset_generation="dataset-1",
            ready=True,
        )
        wire_snapshot = GroupDatasetStateSnapshot(
            rid="rid-1",
            service_id="worker-group-manager-0",
            benchmark_generation=source_snapshot.benchmark_generation,
            dataset_generation=source_snapshot.dataset_generation,
            default_context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
            data_file_path="/tmp/data.bin",
            index_file_path="/tmp/index.bin",
            conversation_count=12,
            total_size_bytes=345,
            ready=source_snapshot.ready,
            error_message=source_snapshot.error_message,
        )

        assert (
            GroupDatasetSnapshot(
                benchmark_generation=wire_snapshot.benchmark_generation,
                dataset_generation=wire_snapshot.dataset_generation,
                ready=wire_snapshot.ready,
                error_message=wire_snapshot.error_message,
            )
            == source_snapshot
        )

    def test_group_lifecycle_contract_exposes_only_group_named_unions(self) -> None:
        """Group-local lifecycle contracts should not retain pod-local aliases."""
        encoder = msgspec.msgpack.Encoder()
        peer_decoder = msgspec.msgpack.Decoder(GroupManagerToPeerMessage)
        manager_decoder = msgspec.msgpack.Decoder(PeerToGroupManagerMessage)
        ready = GroupDatasetReady(
            service_id="worker-group-manager-0",
            data_file_path="/tmp/data.bin",
            index_file_path="/tmp/index.bin",
            conversation_count=4,
            total_size_bytes=128,
        )
        hello = GroupPeerHello(service_id="worker-0", service_type="worker")

        decoded_ready = peer_decoder.decode(encoder.encode(ready))
        decoded_hello = manager_decoder.decode(encoder.encode(hello))

        assert decoded_ready == ready
        assert decoded_hello == hello
        assert not hasattr(CommAddress, "POD_LIFECYCLE")


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

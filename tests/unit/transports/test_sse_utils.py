# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from aiperf.common.enums import SSEFieldType
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.models import SSEField, SSEMessage
from aiperf.transports.sse_utils import AsyncSSEStreamReader


@pytest.fixture
def base_perf_ns() -> int:
    """Fixture providing a base performance counter timestamp."""
    return 1234567890123456789


@pytest.fixture
def expected_empty_message(base_perf_ns: int) -> SSEMessage:
    """Fixture providing an empty SSE message structure."""
    return SSEMessage(perf_ns=base_perf_ns, packets=[])


class TestParseSSEMessage:
    """Comprehensive test suite for SSE message parsing functionality."""

    @pytest.mark.parametrize(
        "raw_message",
        [
            "",  # Completely empty
            "\n",  # Single newline
            "\n\n",  # Multiple newlines
            "   ",  # Only whitespace
            "   \n   \n   ",  # Mixed whitespace and newlines
        ],
    )
    def test_parse_empty_messages(
        self, raw_message: str, base_perf_ns: int, expected_empty_message: SSEMessage
    ) -> None:
        """Test parsing of empty or whitespace-only messages."""
        result = SSEMessage.parse(raw_message, base_perf_ns)
        assert result.perf_ns == expected_empty_message.perf_ns
        assert result.packets == expected_empty_message.packets

    @pytest.mark.parametrize(
        "field_name,field_value,expected_name,expected_value",
        [
            # Standard SSE field types
            ("data", "Hello World", SSEFieldType.DATA, "Hello World"),
            ("event", "message", SSEFieldType.EVENT, "message"),
            ("id", "123456", SSEFieldType.ID, "123456"),
            ("retry", "5000", SSEFieldType.RETRY, "5000"),
            # Custom field names
            ("custom-field", "custom-value", "custom-field", "custom-value"),
            ("X-Custom-Header", "header-value", "X-Custom-Header", "header-value"),
            # Case sensitivity tests (should preserve original case)
            ("Data", "test", "Data", "test"),
            ("EVENT", "test", "EVENT", "test"),
            # Empty values
            ("data", "", SSEFieldType.DATA, ""),
            ("event", "", SSEFieldType.EVENT, ""),
        ],
    )
    def test_parse_single_field_messages(
        self,
        field_name: str,
        field_value: str,
        expected_name: SSEFieldType | str,
        expected_value: str,
        base_perf_ns: int,
    ) -> None:
        """Test parsing of messages with single field."""
        raw_message = f"{field_name}: {field_value}"
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == expected_name
        assert result.packets[0].value == expected_value

    @pytest.mark.parametrize(
        "field_name",
        [
            "data",
            "event",
            "id",
            "retry",
            "custom-field",
        ],
    )
    def test_parse_field_without_colon(
        self, field_name: str, base_perf_ns: int
    ) -> None:
        """Test parsing of fields without colon (no value)."""
        raw_message = field_name
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == field_name
        assert result.packets[0].value is None

    @pytest.mark.parametrize(
        "comment_text",
        [
            "This is a comment",
            "Another comment with special chars: !@#$%^&*()",
            "",  # Empty comment
            "Multi word comment with spaces",
        ],
    )
    def test_parse_comment_messages(self, comment_text: str, base_perf_ns: int) -> None:
        """Test parsing of comment messages (empty field name)."""
        raw_message = f": {comment_text}"
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == SSEFieldType.COMMENT
        assert result.packets[0].value == comment_text

    def test_parse_multiline_complex_message(self, base_perf_ns: int) -> None:
        """Test parsing of complex multi-line SSE message."""
        raw_message = """data: {"message": "Hello"}
event: user-message
id: msg-123
retry: 3000
data: {"continuation": "World"}
: This is a comment
custom-header: custom-value
field-without-value"""

        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 8

        # Verify each field
        expected_fields = [
            SSEField(name=SSEFieldType.DATA, value='{"message": "Hello"}'),
            SSEField(name=SSEFieldType.EVENT, value="user-message"),
            SSEField(name=SSEFieldType.ID, value="msg-123"),
            SSEField(name=SSEFieldType.RETRY, value="3000"),
            SSEField(name=SSEFieldType.DATA, value='{"continuation": "World"}'),
            SSEField(name=SSEFieldType.COMMENT, value="This is a comment"),
            SSEField(name="custom-header", value="custom-value"),
            SSEField(name="field-without-value", value=None),
        ]

        for i, expected_field in enumerate(expected_fields):
            assert result.packets[i].name == expected_field.name
            assert result.packets[i].value == expected_field.value

    @pytest.mark.parametrize(
        "raw_message,expected_packet_count",
        [
            # Multiple data fields
            ("data: line1\ndata: line2\ndata: line3", 3),
            # Mixed fields with empty lines
            ("data: test\n\nevent: msg\n\nid: 123", 3),
            # Comments interspersed
            (": comment1\ndata: test\n: comment2", 3),
            # Fields without values
            ("field1\nfield2\nfield3", 3),
        ],
    )
    def test_parse_multiple_fields(
        self, raw_message: str, expected_packet_count: int, base_perf_ns: int
    ) -> None:
        """Test parsing of messages with multiple fields."""
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == expected_packet_count

    @pytest.mark.parametrize(
        "raw_message,field_name,field_value",
        [
            # Leading/trailing whitespace in field names and values
            ("  data  :  test value  ", SSEFieldType.DATA, "test value"),
            ("\tdata\t:\ttest\t", SSEFieldType.DATA, "test"),
            ("data:value", SSEFieldType.DATA, "value"),  # No spaces around colon
            # Whitespace in field without colon
            ("  field-name  ", "field-name", None),
            # Whitespace in comments
            ("  :  comment text  ", SSEFieldType.COMMENT, "comment text"),
        ],
    )
    def test_parse_whitespace_handling(
        self,
        raw_message: str,
        field_name: str,
        field_value: str | None,
        base_perf_ns: int,
    ) -> None:
        """Test proper handling of whitespace in field parsing."""
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == field_name
        assert result.packets[0].value == field_value

    @pytest.mark.parametrize(
        "colon_content",
        [
            "data: value: with: multiple: colons",
            "data: key:value pairs",
            "data: http://example.com:8000/path",
            "data: time:12:34:56",
        ],
    )
    def test_parse_multiple_colons_in_value(
        self, colon_content: str, base_perf_ns: int
    ) -> None:
        """Test parsing when field values contain multiple colons."""
        result = SSEMessage.parse(colon_content, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == SSEFieldType.DATA

        # Value should be everything after the first colon
        expected_value = colon_content.split(":", 1)[1].strip()
        assert result.packets[0].value == expected_value

    def test_parse_json_data_field(self, base_perf_ns: int) -> None:
        """Test parsing of JSON data in SSE fields."""
        json_data = (
            '{"type": "message", "content": "Hello, World!", "timestamp": 1234567890}'
        )
        raw_message = f"data: {json_data}"

        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == SSEFieldType.DATA
        assert result.packets[0].value == json_data

    def test_parse_real_world_sse_example(self, base_perf_ns: int) -> None:
        """Test parsing of a real-world SSE message example."""
        raw_message = """event: message
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk"}
data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
id: msg_123
retry: 5000"""

        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 5

        # Verify structure
        assert result.packets[0].name == SSEFieldType.EVENT
        assert result.packets[0].value == "message"

        assert result.packets[1].name == SSEFieldType.DATA
        assert '"id": "chatcmpl-123"' in result.packets[1].value  # type: ignore

        assert result.packets[2].name == SSEFieldType.DATA
        assert '"content": "Hello"' in result.packets[2].value  # type: ignore

        assert result.packets[3].name == SSEFieldType.ID
        assert result.packets[3].value == "msg_123"

        assert result.packets[4].name == SSEFieldType.RETRY
        assert result.packets[4].value == "5000"

    @pytest.mark.parametrize(
        "special_chars",
        [
            "unicode: 你好世界",
            "emoji: 🚀💻🎉",
            "special: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "newlines_in_value: line1\\nline2\\nline3",
            "tabs_and_spaces: \t  value  \t",
        ],
    )
    def test_parse_special_characters(
        self, special_chars: str, base_perf_ns: int
    ) -> None:
        """Test parsing of fields containing special characters."""
        raw_message = f"data: {special_chars}"
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].name == SSEFieldType.DATA
        assert result.packets[0].value == special_chars.strip()

    def test_sse_message_inheritance(self, base_perf_ns: int) -> None:
        """Test that SSEMessage properly implements InferenceServerResponse Protocol."""
        result = SSEMessage.parse("data: test", base_perf_ns)

        # Should have the perf_ns field required by InferenceServerResponse Protocol
        assert hasattr(result, "perf_ns")
        assert result.perf_ns == base_perf_ns

        # Should have the packets field specific to SSEMessage
        assert hasattr(result, "packets")
        assert isinstance(result.packets, list)

    def test_sse_field_model_validation(self, base_perf_ns: int) -> None:
        """Test that SSEField models are properly validated."""
        result = SSEMessage.parse("data: test_value", base_perf_ns)

        field = result.packets[0]
        assert isinstance(field, SSEField)
        assert field.name == SSEFieldType.DATA
        assert field.value == "test_value"

        # Test field serialization/validation
        field_dict = field.model_dump()
        assert "name" in field_dict
        assert "value" in field_dict

    @pytest.mark.parametrize(
        "perf_ns_value",
        [
            0,
            1,
            1234567890123456789,
            9223372036854775807,  # max int64
        ],
    )
    def test_perf_ns_values(self, perf_ns_value: int) -> None:
        """Test various perf_ns timestamp values."""
        result = SSEMessage.parse("data: test", perf_ns_value)
        assert result.perf_ns == perf_ns_value

    def test_parse_with_fixture_data(
        self, complex_sse_message_data: dict[str, str], base_perf_ns: int
    ) -> None:
        """Test parsing using complex fixture data."""
        for _, raw_message in complex_sse_message_data.items():
            result = SSEMessage.parse(raw_message, base_perf_ns)

            assert result.perf_ns == base_perf_ns
            assert len(result.packets) > 0

            # Verify all packets are properly formed
            for packet in result.packets:
                assert isinstance(packet, SSEField)
                assert packet.name is not None
                # value can be None for fields without values

    def test_parse_edge_cases(
        self, edge_case_inputs: dict[str, str], base_perf_ns: int
    ) -> None:
        """Test parsing with edge case inputs."""
        for case_name, raw_message in edge_case_inputs.items():
            result = SSEMessage.parse(raw_message, base_perf_ns)

            assert result.perf_ns == base_perf_ns
            assert isinstance(result.packets, list)

            # Some edge cases should produce empty packets lists
            if case_name in [
                "empty_string",
                "only_newlines",
                "only_whitespace",
                "mixed_whitespace",
            ]:
                assert len(result.packets) == 0
            else:
                assert len(result.packets) > 0

    @pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
    def test_parse_different_line_endings(
        self, line_ending: str, base_perf_ns: int
    ) -> None:
        """Test parsing with different line ending styles."""
        # Note: The current implementation splits on "\n" only
        # This test documents the current behavior
        lines = ["data: line1", "data: line2", "data: line3"]
        raw_message = line_ending.join(lines)

        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert result.perf_ns == base_perf_ns

        if line_ending == "\n":
            # Standard newlines should work
            assert len(result.packets) == 3
        else:
            # Other line endings might not be split properly
            # This documents current behavior - could be enhanced in future
            assert len(result.packets) >= 1

    def test_parse_sse_message_immutability(self, base_perf_ns: int) -> None:
        """Test that parsing produces immutable-like results."""
        raw_message = "data: test"
        result1 = SSEMessage.parse(raw_message, base_perf_ns)
        result2 = SSEMessage.parse(raw_message, base_perf_ns)

        # Should produce equivalent but separate objects
        assert result1.perf_ns == result2.perf_ns
        assert len(result1.packets) == len(result2.packets)
        assert result1.packets[0].name == result2.packets[0].name
        assert result1.packets[0].value == result2.packets[0].value

        # But should be separate objects
        assert result1 is not result2
        assert result1.packets is not result2.packets

    def test_sse_field_type_enum_usage(self, base_perf_ns: int) -> None:
        """Test that SSEFieldType enum is used correctly."""
        test_cases = [
            ("data", SSEFieldType.DATA),
            ("event", SSEFieldType.EVENT),
            ("id", SSEFieldType.ID),
            ("retry", SSEFieldType.RETRY),
        ]

        for field_str, expected_enum in test_cases:
            raw_message = f"{field_str}: test"
            result = SSEMessage.parse(raw_message, base_perf_ns)

            assert result.packets[0].name == expected_enum
            assert str(result.packets[0].name) == field_str

    def test_comment_field_special_handling(self, base_perf_ns: int) -> None:
        """Test special handling of comment fields (empty field name)."""
        raw_message = ": this is a comment"
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert len(result.packets) == 1
        assert result.packets[0].name == SSEFieldType.COMMENT
        assert result.packets[0].value == "this is a comment"

    @pytest.mark.parametrize(
        "field_name_case", ["data", "DATA", "Data", "dAtA", "Event", "ID", "RETRY"]
    )
    def test_case_sensitivity_preservation(
        self, field_name_case: str, base_perf_ns: int
    ) -> None:
        """Test that field name case is preserved exactly as provided."""
        raw_message = f"{field_name_case}: test"
        result = SSEMessage.parse(raw_message, base_perf_ns)

        assert len(result.packets) == 1
        # The name should match exactly what was provided
        if field_name_case.lower() in ["data", "event", "id", "retry"]:
            # For standard fields, the enum should handle case insensitivity
            expected_enum = getattr(SSEFieldType, field_name_case.upper())
            assert result.packets[0].name == expected_enum
        else:
            # For non-standard fields, preserve exact case
            assert result.packets[0].name == field_name_case


class TestParseSSEMessagePerformance:
    @pytest.mark.performance
    def test_parse_large_message(self, base_perf_ns: int) -> None:
        """Test parsing performance with large messages."""
        # Create a large SSE message with many fields
        large_data = "x" * 10000  # 10KB of data
        raw_message = f"data: {large_data}"

        start_time = time.perf_counter()
        result = SSEMessage.parse(raw_message, base_perf_ns)
        end_time = time.perf_counter()

        # Should parse quickly (less than 100ms for 10KB)
        assert (end_time - start_time) < 0.1
        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1
        assert result.packets[0].value == large_data

    @pytest.mark.performance
    def test_parse_many_packets(self, base_perf_ns: int) -> None:
        """Test parsing performance with many fields."""
        # Create message with 1000 fields
        lines = [f"data: field_{i}_data" for i in range(1000)]
        raw_message = "\n".join(lines)

        start_time = time.perf_counter()
        result = SSEMessage.parse(raw_message, base_perf_ns)
        end_time = time.perf_counter()

        # Should parse quickly (less than 250ms for 1000 fields)
        assert (end_time - start_time) < 0.250
        assert result.perf_ns == base_perf_ns
        assert len(result.packets) == 1000


class TestInspectMessageForError:
    """Test suite for SSE error message inspection functionality."""

    @pytest.mark.parametrize(
        "raw_message",
        [
            'data: {"content": "Hello World"}\nevent: message\nid: msg_123',
            "event: message\n: This is a comment\ndata: content",
            ": This is just a comment",
            "",
        ],
    )
    def test_non_error_messages_pass_through(
        self, raw_message: str, base_perf_ns: int
    ) -> None:
        """Test that messages without error events pass through without raising."""
        message = SSEMessage.parse(raw_message, base_perf_ns)
        AsyncSSEStreamReader.inspect_message_for_error(message)

    @pytest.mark.parametrize(
        "raw_message,expected_error_text",
        [
            (
                "event: error\n: Authentication failed\ndata: {}",
                "Authentication failed",
            ),
            (
                "event: error\n: Rate limit exceeded\ndata: {}",
                "Rate limit exceeded",
            ),
            (
                "event: message\nevent: error\n: Multiple events error\ndata: {}",
                "Multiple events error",
            ),
            (
                "event: error\n: 你好世界 🚀 !@#$%^&*()\ndata: {}",
                "你好世界 🚀 !@#$%^&*()",
            ),
        ],
    )  # fmt: skip
    def test_error_event_with_comment(
        self, raw_message: str, expected_error_text: str, base_perf_ns: int
    ) -> None:
        """Test that error events with comments raise SSEResponseError with comment message."""
        message = SSEMessage.parse(raw_message, base_perf_ns)

        with pytest.raises(SSEResponseError) as exc_info:
            AsyncSSEStreamReader.inspect_message_for_error(message)

        assert expected_error_text in str(exc_info.value)
        assert exc_info.value.error_code == 502

    def test_error_event_without_comment(self, base_perf_ns: int) -> None:
        """Test that error event without comment raises with full message."""
        raw_message = 'event: error\ndata: {"error": "Something went wrong"}'
        message = SSEMessage.parse(raw_message, base_perf_ns)

        with pytest.raises(SSEResponseError) as exc_info:
            AsyncSSEStreamReader.inspect_message_for_error(message)

        assert "Unknown error in SSE response" in str(exc_info.value)
        assert exc_info.value.error_code == 502

    def test_error_event_with_multiple_comments_uses_first(
        self, base_perf_ns: int
    ) -> None:
        """Test that when multiple comment fields exist, first one is used."""
        raw_message = "event: error\n: First error\n: Second error\ndata: {}"
        message = SSEMessage.parse(raw_message, base_perf_ns)

        with pytest.raises(SSEResponseError) as exc_info:
            AsyncSSEStreamReader.inspect_message_for_error(message)

        assert "First error" in str(exc_info.value)
        assert "Second error" not in str(exc_info.value)

    @pytest.mark.parametrize("event_case", ["error", "ERROR", "Error", "eRrOr"])
    def test_error_event_case_insensitive(
        self, event_case: str, base_perf_ns: int
    ) -> None:
        """Test that error event detection is case-insensitive."""
        raw_message = f"event: {event_case}\n: Error message"
        message = SSEMessage.parse(raw_message, base_perf_ns)

        with pytest.raises(SSEResponseError) as exc_info:
            AsyncSSEStreamReader.inspect_message_for_error(message)

        assert "Error message" in str(exc_info.value)
        assert exc_info.value.error_code == 502


@pytest.fixture
def create_mock_sse_iterator():
    """Factory fixture for creating mock SSE async iterators."""

    def _factory(*chunks: bytes):
        async def mock_async_iter():
            for chunk in chunks:
                yield chunk

        return mock_async_iter()

    return _factory


class TestAsyncSSEStreamReaderErrorHandling:
    """Test suite for AsyncSSEStreamReader integration with error handling."""

    async def test_read_complete_stream_success_no_errors(
        self, create_mock_sse_iterator
    ) -> None:
        """Test that read_complete_stream completes successfully when no errors."""
        reader = AsyncSSEStreamReader(
            create_mock_sse_iterator(
                b"data: Hello\n\n",
                b"event: message\ndata: World\n\n",
                b"data: [DONE]\n\n",
            )
        )
        messages = await reader.read_complete_stream()

        assert len(messages) == 3
        assert all(isinstance(msg, SSEMessage) for msg in messages)

    @pytest.mark.parametrize(
        "chunks,expected_error,expected_msg_count",
        [
            (
                (b"event: error\n: Rate limit exceeded\n\n",),
                "Rate limit exceeded",
                0,
            ),
            (
                (
                    b"data: Message 1\n\n",
                    b"event: error\n: Server overloaded\n\n",
                ),
                "Server overloaded",
                1,
            ),
            (
                (
                    b"data: Hello\n\n",
                    b"event: error\n: Connection timeout\ndata: {}\n\n",
                    b"data: Should not reach\n\n",
                ),
                "Connection timeout",
                1,
            ),
            (
                (
                    b'data: {"content": "Hello"}\n\n',
                    b'event: message\ndata: {"content": "World"}\n\n',
                    b"event: error\n: Authentication expired\ndata: {}\n\n",
                ),
                "Authentication expired",
                2,
            ),
        ],
    )
    async def test_stream_errors_at_various_positions(
        self,
        create_mock_sse_iterator,
        chunks: tuple[bytes, ...],
        expected_error: str,
        expected_msg_count: int,
    ) -> None:
        """Test error handling at different positions in the stream."""
        reader = AsyncSSEStreamReader(create_mock_sse_iterator(*chunks))

        if expected_msg_count == 0:
            # For read_complete_stream, error is raised immediately
            with pytest.raises(SSEResponseError) as exc_info:
                await reader.read_complete_stream()
            assert expected_error in str(exc_info.value)
        else:
            # For manual iteration, we can count messages before error
            messages = []
            with pytest.raises(SSEResponseError) as exc_info:
                async for message in reader:
                    AsyncSSEStreamReader.inspect_message_for_error(message)
                    messages.append(message)

            assert len(messages) == expected_msg_count
            assert expected_error in str(exc_info.value)

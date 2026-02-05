# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from aiperf.common.config.config_validators import (
    coerce_value,
    parse_str_as_numeric_dict,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list_of_positive_values,
)


class TestCoerceValue:
    """Test suite for the coerce_value function."""

    @pytest.mark.parametrize(
        "input,expected",
        [
            ("1", 1),
            ("1.0", 1.0),
            ("true", True),
            ("false", False),
            ("none", None),
            ("null", None),
            ("0", 0),
            ("0.0", 0.0),
            ("-0.0", 0.0),
            (".5", 0.5),
            ("0.5", 0.5),
            ("-1", -1),
            ("-1.0", -1.0),
            ("-1.5", -1.5),
            ("Hello", "Hello"),
            ("", ""),
            ("NONE", None),
            ("NULL", None),
            ("True", True),
            ("False", False),
            ("-5s", "-5s"),
            ("-5.0s", "-5.0s"),
            ("127.0.0.1", "127.0.0.1"),
            ("127.0.0.1:8000", "127.0.0.1:8000"),
            ("a.b", "a.b"),
            ("a.b:c.d", "a.b:c.d"),
            (".b", ".b"),
            ("-.b", "-.b"),
            ("-.5", -0.5),
            ("-0.5", -0.5),
            ("32.b", "32.b"),
            ("-0", 0),
            ("0123", "0123"),
            ("-0123", "-0123"),
            ("0.0123", 0.0123),
        ],
    )
    def test_coerce_value(self, input: Any, expected: Any) -> None:
        assert coerce_value(input) == expected


class TestParseStrOrDictAsTupleList:
    """Test suite for the parse_str_or_dict_as_tuple_list function."""

    def test_empty_dict_input(self):
        """Test that empty dict input is returned unchanged."""
        result = parse_str_or_dict_as_tuple_list({})
        assert result == []

    @pytest.mark.parametrize(
        "input_list,expected",
        [
            (
                ["key1:value1", "key2:value2", "key3:false"],
                [("key1", "value1"), ("key2", "value2"), ("key3", False)],
            ),
            (["name:John", "age:30"], [("name", "John"), ("age", 30)]),
            (
                ["  key1  :  value1  ", "key2:value2"],
                [("key1", "value1"), ("key2", "value2")],
            ),
            (["single:item"], [("single", "item")]),
        ],
    )
    def test_list_input_converts_to_dict(self, input_list, expected):
        """Test that list input is converted to dict by splitting on colons."""
        result = parse_str_or_dict_as_tuple_list(input_list)
        assert result == expected

    def test_empty_list_input(self):
        """Test that empty list input returns empty dict."""
        result = parse_str_or_dict_as_tuple_list([])
        assert result == []

    @pytest.mark.parametrize(
        "json_string,expected",
        [
            (
                '{"key1": "value1", "key2": "value2"}',
                [("key1", "value1"), ("key2", "value2")],
            ),
            ('{"name": "John", "age": 30}', [("name", "John"), ("age", 30)]),
            ('{"nested": {"key": "value"}}', [("nested", {"key": "value"})]),
            ('{"empty": {}}', [("empty", {})]),
            ("{}", []),
        ],
    )
    def test_json_string_input_parses_correctly(self, json_string, expected):
        """Test that JSON string input is parsed correctly."""
        result = parse_str_or_dict_as_tuple_list(json_string)
        assert result == expected

    @pytest.mark.parametrize(
        "comma_separated_string,expected",
        [
            ("key1:value1,key2:value2", [("key1", "value1"), ("key2", "value2")]),
            ("name:John,age:30", [("name", "John"), ("age", 30)]),
            (
                "  key1  :  value1  ,  key2  :  value2  ",
                [("key1", "value1"), ("key2", "value2")],
            ),
            ("single:item", [("single", "item")]),
        ],
    )
    def test_comma_separated_string_input_converts_to_dict(
        self, comma_separated_string, expected
    ):
        """Test that comma-separated string input is converted to dict."""
        result = parse_str_or_dict_as_tuple_list(comma_separated_string)
        assert result == expected

    def test_empty_string_input(self):
        """Test that empty string input raises ValueError."""
        with pytest.raises(ValueError):
            parse_str_or_dict_as_tuple_list("")

    @pytest.mark.parametrize(
        "invalid_json",
        [
            '{"key1": "value1", "key2":}',  # Missing value
            '{"key1": "value1" "key2": "value2"}',  # Missing comma
            '{"key1": "value1",}',  # Trailing comma
            '{key1: "value1"}',  # Unquoted key
            '{"key1": value1}',  # Unquoted value
            "{invalid json}",  # Invalid JSON
        ],
    )
    def test_invalid_json_string_raises_value_error(self, invalid_json):
        """Test that invalid JSON string raises ValueError."""
        with pytest.raises(ValueError, match="must be a valid JSON string"):
            parse_str_or_dict_as_tuple_list(invalid_json)

    @pytest.mark.parametrize(
        "invalid_list",
        [
            ["key1_no_colon"],  # Missing colon
            ["key1:value1", "key2_no_colon"],  # One valid, one invalid
        ],
    )
    def test_invalid_list_format_raises_value_error(self, invalid_list):
        """Test that list with invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_str_or_dict_as_tuple_list(invalid_list)

    @pytest.mark.parametrize(
        "invalid_string",
        [
            "key1_no_colon",  # Missing colon
            "key1:value1,key2_no_colon",  # One valid, one invalid
        ],
    )
    def test_invalid_string_format_raises_value_error(self, invalid_string):
        """Test that string with invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_str_or_dict_as_tuple_list(invalid_string)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            123,  # Integer
            12.34,  # Float
            True,  # Boolean
            object(),  # Object
        ],
    )
    def test_invalid_input_type_raises_value_error(self, invalid_input):
        """Test that invalid input types raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid string, list, or dict"):
            parse_str_or_dict_as_tuple_list(invalid_input)

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            # String with multiple colons
            (
                "key1:value1:extra,key2:value2",
                [("key1", "value1:extra"), ("key2", "value2")],
            ),
            # List with multiple colons
            (
                ["key1:value1:extra", "key2:value2"],
                [("key1", "value1:extra"), ("key2", "value2")],
            ),
            # URL with port
            ("url:http://example.com:8080", [("url", "http://example.com:8080")]),
            # Multiple entries with colons in values (timestamps, ports, etc)
            (
                "server:localhost:8080,time:12:30:45,status:active",
                [
                    ("server", "localhost:8080"),
                    ("time", "12:30:45"),
                    ("status", "active"),
                ],
            ),
        ],
    )
    def test_values_can_contain_colons(self, input_value, expected):
        """Test that values can contain colons (URLs, timestamps, etc)."""
        result = parse_str_or_dict_as_tuple_list(input_value)
        assert result == expected

    def test_whitespace_handling_in_string_input(self):
        """Test that whitespace is properly trimmed in string input."""
        result = parse_str_or_dict_as_tuple_list(
            "  key1  :  value1  ,  key2  :  value2  "
        )
        expected = [("key1", "value1"), ("key2", "value2")]
        assert result == expected

    def test_whitespace_handling_in_list_input(self):
        """Test that whitespace is properly trimmed in list input."""
        result = parse_str_or_dict_as_tuple_list(
            ["  key1  :  value1  ", "  key2  :  value2  "]
        )
        expected = [("key1", "value1"), ("key2", "value2")]
        assert result == expected

    def test_json_string_with_complex_data_types(self):
        """Test JSON string with complex data types."""
        complex_json = '{"string": "value", "number": 42, "boolean": true, "null": null, "array": [1, 2, 3]}'
        result = parse_str_or_dict_as_tuple_list(complex_json)
        expected = [
            ("string", "value"),
            ("number", 42),
            ("boolean", True),
            ("null", None),
            ("array", [1, 2, 3]),
        ]
        assert result == expected

    def test_error_message_contains_input_for_invalid_json(self):
        """Test that error message contains the input for invalid JSON."""
        invalid_json = '{"invalid": json}'
        with pytest.raises(ValueError) as exc_info:
            parse_str_or_dict_as_tuple_list(invalid_json)
        assert invalid_json in str(exc_info.value)

    def test_error_message_contains_input_for_invalid_type(self):
        """Test that error message contains the input for invalid types."""
        invalid_input = 123
        with pytest.raises(ValueError) as exc_info:
            parse_str_or_dict_as_tuple_list(invalid_input)
        assert "123" in str(exc_info.value)

    def test_none_input_returns_none(self):
        """Test that none input returns none."""
        result = parse_str_or_dict_as_tuple_list(None)
        assert result is None

    @pytest.mark.parametrize(
        "input_list,expected",
        [
            (
                [["temperature", 0.1], ["max_tokens", 150]],
                [("temperature", 0.1), ("max_tokens", 150)],
            ),
            (
                [("temperature", 0.1), ("max_tokens", 150)],
                [("temperature", 0.1), ("max_tokens", 150)],
            ),
            (
                [("key1", "value1"), ("key2", 123), ("key3", True)],
                [("key1", "value1"), ("key2", 123), ("key3", True)],
            ),
        ],
    )
    def test_list_of_key_value_pairs_input(self, input_list, expected):
        """Test that a list of key-value pairs (lists/tuples) is converted correctly to a list of tuples."""
        result = parse_str_or_dict_as_tuple_list(input_list)
        assert result == expected
        # Make sure that the result is the same when parsed again.
        result2 = parse_str_or_dict_as_tuple_list(result)
        assert result2 == expected


class TestParseStrOrListOfPositiveValues:
    """Test suite for the parse_str_or_list_of_positive_values function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("1,2,3", [1, 2, 3]),
            ([1, 2, 3], [1, 2, 3]),
            (["1", "2", "3"], [1, 2, 3]),
            ("1.5,2.0,3.25", [1.5, 2.0, 3.25]),
            (["1.5", "2.0", "3.25"], [1.5, 2.0, 3.25]),
            ([1.5, 2.0, 3.25], [1.5, 2.0, 3.25]),
            (["1", "2.5", "3"], [1, 2.5, 3]),
            ("1e2,2e2", [100.0, 200.0]),
            (["1e2", "2e2"], [100.0, 200.0]),
            (["1.0", "1e2", "2.5"], [1.0, 100.0, 2.5]),
        ],
    )
    def test_valid_inputs(self, input_value, expected):
        result = parse_str_or_list_of_positive_values(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "0,-1,2",  # Zero and negative
            [0, 1, 2],  # Zero
            [-1, 2, 3],  # Negative
            ["-1", "2", "3"],  # Negative string
            "a,b,c",  # Non-numeric
            ["1", "foo", "3"],  # Mixed valid/invalid
        ],
    )
    def test_invalid_inputs_raise_value_error(self, invalid_input):
        with pytest.raises(ValueError):
            parse_str_or_list_of_positive_values(invalid_input)

    def test_parse_str_as_numeric_dict_simple(self):
        assert parse_str_as_numeric_dict(
            "request_latency:250 inter_token_latency:10"
        ) == {
            "request_latency": 250.0,
            "inter_token_latency": 10.0,
        }

    @pytest.mark.parametrize(
        "error_message,pattern",
        [
            ("", "expected space-separated 'key:value' pairs"),
            ("   ", "expected space-separated 'key:value' pairs"),
            (123, "expected a string"),
            ("request_latency250", "not in 'key:value'"),
            ("a:", "empty value"),
            (":1", "empty key"),
            ("a:b", "must be numeric"),
            ("a:1 badpair b:2", "not in 'key:value'"),
        ],
    )
    def test_parse_str_as_numeric_dict_error_param(self, error_message, pattern):
        with pytest.raises(ValueError, match=pattern):
            parse_str_as_numeric_dict(error_message)

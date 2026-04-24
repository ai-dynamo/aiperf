# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Message.from_json tagged-union routing."""

import json

import msgspec
import pytest

from aiperf.common.enums import (
    LifecycleState,
    MessageType,
)
from aiperf.common.messages import Message, StatusMessage


def assert_routed_to(msg, expected_class, **expected_attrs):
    """Assert message routed to expected class with expected attributes."""
    assert isinstance(msg, expected_class), (
        f"Expected {expected_class.__name__}, got {type(msg).__name__}"
    )
    for attr, value in expected_attrs.items():
        assert getattr(msg, attr) == value, (
            f"Expected {attr}={value}, got {getattr(msg, attr)}"
        )


class TestMessageRouting:
    """Test Message.from_json routing behavior."""

    @pytest.mark.parametrize(
        "data,expected_class,expected_attrs",
        [
            # Single-level routing
            (
                {
                    "message_type": "status",
                    "state": "running",
                    "service_id": "test-service",
                    "service_type": "worker",
                },
                StatusMessage,
                {"message_type": MessageType.STATUS, "state": LifecycleState.RUNNING},
            ),
        ],
    )  # fmt: skip
    def test_routing_levels(self, data, expected_class, expected_attrs):
        """Test routing at various nesting levels."""
        msg = Message.from_json(data)
        assert_routed_to(msg, expected_class, **expected_attrs)

    def test_json_string_routing(self, base_message_data):
        """Test routing from JSON string (ensures single parse)."""
        data = {
            **base_message_data,
            "message_type": "status",
            "state": "running",
            "service_type": "worker",
        }
        msg = Message.from_json(json.dumps(data))
        assert_routed_to(msg, StatusMessage, state=LifecycleState.RUNNING)

    @pytest.mark.parametrize(
        "data,match",
        [
            ({"service_id": "test"}, "missing required field `message_type`"),
        ],
    )  # fmt: skip
    def test_missing_discriminator_error(self, data, match):
        """Test that missing discriminators raise ValidationError."""
        with pytest.raises(msgspec.ValidationError, match=match):
            Message.from_json(data)

    def test_unknown_discriminator_value_raises(self):
        """Test that unknown discriminator values raise ValidationError.

        msgspec is strict — decode fails fast for unknown message_type tags.
        """
        data = {
            "message_type": "unknown_type",
            "service_id": "test",
        }
        with pytest.raises(msgspec.ValidationError):
            Message.from_json(data)

    @pytest.mark.parametrize(
        "input_transform,description",
        [
            (lambda d: d, "dict (no parsing)"),
            (lambda d: json.dumps(d), "JSON string"),
            (lambda d: json.dumps(d).encode("utf-8"), "bytes"),
            (lambda d: bytearray(json.dumps(d).encode("utf-8")), "bytearray"),
        ],
    )  # fmt: skip
    def test_from_json_input_types(self, input_transform, description):
        """Test that from_json accepts various input types: dict, str, bytes, bytearray."""
        data = {
            "message_type": "status",
            "state": "running",
            "service_id": "test-service",
            "service_type": "worker",
        }
        msg = Message.from_json(input_transform(data))
        assert_routed_to(msg, StatusMessage, state=LifecycleState.RUNNING)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-request extra field on SingleTurn and MooncakeTrace row models."""

import pytest
from pytest import param

from aiperf.dataset.loader.models import MooncakeTrace, SingleTurn


class TestSingleTurnExtraField:
    """Tests for the extra field on SingleTurn."""

    def test_extra_field_absent_defaults_to_none(self):
        """extra defaults to None when omitted."""
        row = SingleTurn(text="hello")
        assert row.extra is None

    def test_extra_field_present(self):
        """extra is stored when provided."""
        row = SingleTurn(text="hello", extra={"nvext": {"priority": 1}})
        assert row.extra == {"nvext": {"priority": 1}}

    def test_extra_field_none_explicit(self):
        """extra accepts explicit None."""
        row = SingleTurn(text="hello", extra=None)
        assert row.extra is None

    def test_extra_serialization_round_trip(self):
        """extra survives JSON serialization and back."""
        row = SingleTurn(text="hello", extra={"key": "value", "num": 42})
        json_str = row.model_dump_json()
        restored = SingleTurn.model_validate_json(json_str)
        assert restored.extra == {"key": "value", "num": 42}

    def test_extra_round_trip_absent(self):
        """Round-trip with extra absent preserves None."""
        row = SingleTurn(text="hello")
        json_str = row.model_dump_json()
        restored = SingleTurn.model_validate_json(json_str)
        assert restored.extra is None

    @pytest.mark.parametrize(
        "extra",
        [
            param({"nvext": {"priority": 1}}, id="nested_dict"),
            param({"key": "value"}, id="flat_string_value"),
            param({"num": 42, "flag": True}, id="mixed_types"),
            param({}, id="empty_dict"),
        ],
    )  # fmt: skip
    def test_extra_accepts_various_dict_shapes(self, extra):
        """extra accepts various dict shapes."""
        row = SingleTurn(text="hello", extra=extra)
        assert row.extra == extra


class TestMooncakeTraceExtraField:
    """Tests for the extra field on MooncakeTrace."""

    def test_extra_field_absent_defaults_to_none(self):
        """extra defaults to None when omitted."""
        trace = MooncakeTrace(input_length=100)
        assert trace.extra is None

    def test_extra_field_present(self):
        """extra is stored when provided."""
        trace = MooncakeTrace(input_length=100, extra={"nvext": {"priority": 2}})
        assert trace.extra == {"nvext": {"priority": 2}}

    def test_extra_field_with_messages_input(self):
        """extra works alongside the messages input mode."""
        messages = [{"role": "user", "content": "Hello"}]
        trace = MooncakeTrace(messages=messages, extra={"routing": "fast"})
        assert trace.extra == {"routing": "fast"}

    def test_extra_serialization_round_trip(self):
        """extra survives JSON serialization and back."""
        trace = MooncakeTrace(input_length=50, extra={"k": "v"})
        json_str = trace.model_dump_json()
        restored = MooncakeTrace.model_validate_json(json_str)
        assert restored.extra == {"k": "v"}

    def test_extra_round_trip_absent(self):
        """Round-trip with extra absent preserves None."""
        trace = MooncakeTrace(input_length=50)
        json_str = trace.model_dump_json()
        restored = MooncakeTrace.model_validate_json(json_str)
        assert restored.extra is None

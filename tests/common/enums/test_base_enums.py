# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class SampleEnum(CaseInsensitiveStrEnum):
    """Sample enum for CaseInsensitiveStrEnum tests."""

    OPTION_ONE = "option_one"
    OPTION_TWO = "option_two"
    MULTI_WORD = "multi_word_option"


class TestCaseInsensitiveStrEnum:
    """Test class for CaseInsensitiveStrEnum functionality."""

    @pytest.mark.parametrize(
        "enum_member,string_value",
        [
            (SampleEnum.OPTION_ONE, "option-one"),
            (SampleEnum.OPTION_ONE, "OPTION-ONE"),
            (SampleEnum.OPTION_ONE, "option_one"),
            (SampleEnum.OPTION_ONE, "OPTION_ONE"),
            (SampleEnum.OPTION_ONE, "OpTiOn-OnE"),
            (SampleEnum.OPTION_TWO, "option-two"),
            (SampleEnum.OPTION_TWO, "OPTION_TWO"),
            (SampleEnum.MULTI_WORD, "multi-word-option"),
            (SampleEnum.MULTI_WORD, "MULTI_WORD_OPTION"),
            (SampleEnum.MULTI_WORD, "multi_word-option"),
        ],
    )
    def test_case_insensitive_equality(self, enum_member, string_value):
        """Test case-insensitive equality with underscore/hyphen normalization."""
        assert enum_member == string_value

    @pytest.mark.parametrize(
        "value,expected_member",
        [
            ("option-one", SampleEnum.OPTION_ONE),
            ("OPTION_ONE", SampleEnum.OPTION_ONE),
            ("Option-One", SampleEnum.OPTION_ONE),
            ("option_two", SampleEnum.OPTION_TWO),
            ("OPTION-TWO", SampleEnum.OPTION_TWO),
            ("multi-word-option", SampleEnum.MULTI_WORD),
            ("MULTI_WORD_OPTION", SampleEnum.MULTI_WORD),
            ("multi_word-option", SampleEnum.MULTI_WORD),
        ],
    )
    def test_missing_case_insensitive_lookup(self, value, expected_member):
        """Test that _missing_ enables flexible enum construction from various input formats."""
        result = SampleEnum(value)
        assert result is expected_member

    @pytest.mark.parametrize(
        "enum_member,expected_str",
        [
            (SampleEnum.OPTION_ONE, "option_one"),
            (SampleEnum.OPTION_TWO, "option_two"),
            (SampleEnum.MULTI_WORD, "multi_word_option"),
        ],
    )
    def test_str_returns_original_value(self, enum_member, expected_str):
        """Test that __str__ returns the original enum value unchanged."""
        assert str(enum_member) == expected_str
        assert str(enum_member) == enum_member.value
        assert enum_member.value == expected_str

    @pytest.mark.parametrize(
        "enum_member,expected_repr",
        [
            (SampleEnum.OPTION_ONE, "SampleEnum.OPTION_ONE"),
            (SampleEnum.OPTION_TWO, "SampleEnum.OPTION_TWO"),
            (SampleEnum.MULTI_WORD, "SampleEnum.MULTI_WORD"),
        ],
    )
    def test_repr_format(self, enum_member, expected_repr):
        """Test that __repr__ returns EnumClass.MEMBER_NAME format."""
        assert repr(enum_member) == expected_repr

    @pytest.mark.parametrize(
        "enum_member,expected_normalized",
        [
            (SampleEnum.OPTION_ONE, "option-one"),
            (SampleEnum.OPTION_TWO, "option-two"),
            (SampleEnum.MULTI_WORD, "multi-word-option"),
        ],
    )
    def test_normalized_value_property(self, enum_member, expected_normalized):
        """Test that normalized_value returns lowercase with hyphens."""
        assert enum_member.normalized_value == expected_normalized

    def test_hash_based_on_normalized_value(self):
        """Test that hash is based on normalized value, critical for dict/set operations."""
        enum_member = SampleEnum.OPTION_ONE
        assert hash(enum_member) == hash(enum_member.normalized_value)

        test_dict = {SampleEnum.OPTION_ONE: "value"}
        lookup_member = SampleEnum("option-one")
        assert test_dict[lookup_member] == "value"

    def test_enum_to_enum_comparison(self):
        """Test enum-to-enum comparison, including cross-enum comparison."""
        assert SampleEnum.OPTION_ONE == SampleEnum.OPTION_ONE
        assert SampleEnum.OPTION_ONE != SampleEnum.OPTION_TWO

        class DifferentEnum(CaseInsensitiveStrEnum):
            OPTION_ONE = "option_one"

        # Cross-enum comparison works due to normalized value comparison
        assert SampleEnum.OPTION_ONE == DifferentEnum.OPTION_ONE
        assert SampleEnum.OPTION_ONE is not DifferentEnum.OPTION_ONE

    @pytest.mark.parametrize(
        "enum_member,non_matching",
        [
            (SampleEnum.OPTION_ONE, "something_else"),
            (SampleEnum.OPTION_ONE, ""),
            (SampleEnum.OPTION_ONE, "  option_one  "),
            (SampleEnum.OPTION_ONE, "option one"),
            (SampleEnum.OPTION_ONE, 123),
            (SampleEnum.OPTION_ONE, None),
            (SampleEnum.OPTION_ONE, []),
        ],
    )
    def test_inequality_with_non_matching_values(self, enum_member, non_matching):
        """Test inequality with various non-matching values and types."""
        assert enum_member != non_matching

    def test_invalid_value_raises_error(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError, match="is not a valid"):
            SampleEnum("invalid_option")

    def test_multiple_consecutive_separators_rejected(self):
        """Test that multiple consecutive separators are preserved."""
        with pytest.raises(ValueError):
            SampleEnum("option__one")

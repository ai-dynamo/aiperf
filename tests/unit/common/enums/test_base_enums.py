# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CaseInsensitiveStrEnum and dash/underscore normalization."""

import pytest
from pydantic import BaseModel, ValidationError

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum, _normalize_name

# =============================================================================
# Fixtures
# =============================================================================


class SampleEnum(CaseInsensitiveStrEnum):
    """Sample enum for testing."""

    ALPHA = "alpha"
    BETA = "beta"
    FOO_BAR = "foo_bar"


class DashValueEnum(CaseInsensitiveStrEnum):
    """Enum with dash values (CLI convention)."""

    MY_VALUE = "my-value"
    OTHER_VALUE = "other-value"


# =============================================================================
# _normalize_name Tests
# =============================================================================


class TestNormalizeName:
    """Tests for _normalize_name helper function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("foo_bar", "foo_bar"),
            ("foo-bar", "foo_bar"),
            ("FOO_BAR", "foo_bar"),
            ("FOO-BAR", "foo_bar"),
            ("Foo_Bar", "foo_bar"),
            ("Foo-Bar", "foo_bar"),
            ("foo--bar", "foo__bar"),
            ("foo__bar", "foo__bar"),
            ("", ""),
        ],
    )  # fmt: skip
    def test_normalize_name(self, input_value, expected):
        """_normalize_name converts to lowercase and replaces dashes with underscores."""
        assert _normalize_name(input_value) == expected


# =============================================================================
# Basic CaseInsensitiveStrEnum Tests
# =============================================================================


class TestCaseInsensitiveStrEnum:
    """Tests for basic CaseInsensitiveStrEnum functionality."""

    def test_str_returns_value(self):
        """str() returns the enum value."""
        assert str(SampleEnum.ALPHA) == "alpha"

    def test_repr_format(self):
        """repr() returns ClassName.MEMBER format."""
        assert repr(SampleEnum.ALPHA) == "SampleEnum.ALPHA"

    def test_is_str_subclass(self):
        """Enum members are str subclass."""
        assert isinstance(SampleEnum.ALPHA, str)
        assert SampleEnum.ALPHA.upper() == "ALPHA"


# =============================================================================
# Case-Insensitive Lookup Tests
# =============================================================================


class TestCaseInsensitiveLookup:
    """Tests for case-insensitive enum construction and comparison."""

    @pytest.mark.parametrize(
        "input_value,expected_member",
        [
            ("alpha", SampleEnum.ALPHA),
            ("ALPHA", SampleEnum.ALPHA),
            ("Alpha", SampleEnum.ALPHA),
            ("aLpHa", SampleEnum.ALPHA),
        ],
    )  # fmt: skip
    def test_construction_case_insensitive(self, input_value, expected_member):
        """Enum construction is case-insensitive."""
        assert SampleEnum(input_value) == expected_member

    @pytest.mark.parametrize(
        "compare_value,expected",
        [
            ("alpha", True),
            ("ALPHA", True),
            ("Alpha", True),
            ("beta", False),
        ],
    )  # fmt: skip
    def test_eq_case_insensitive(self, compare_value, expected):
        """__eq__ is case-insensitive for strings."""
        assert (compare_value == SampleEnum.ALPHA) == expected

    def test_eq_enum_members(self):
        """Same member equals itself, different members don't."""
        assert SampleEnum.ALPHA == SampleEnum.ALPHA
        assert SampleEnum.ALPHA != SampleEnum.BETA

    def test_eq_non_string_returns_false(self):
        """__eq__ returns False for non-string, non-enum types."""
        assert SampleEnum.ALPHA != 123
        assert SampleEnum.ALPHA != None  # noqa: E711 - intentionally testing __eq__ with None
        assert SampleEnum.ALPHA != []


# =============================================================================
# Dash/Underscore Normalization Tests
# =============================================================================


class TestDashUnderscoreNormalization:
    """Tests for dash/underscore normalization in enum operations."""

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo_Bar", "Foo-Bar"],
    )  # fmt: skip
    def test_construction_normalizes_dashes(self, input_value):
        """Construction normalizes dashes to underscores."""
        result = SampleEnum(input_value)
        assert result == SampleEnum.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["my_value", "my-value", "MY_VALUE", "MY-VALUE"],
    )  # fmt: skip
    def test_construction_with_dash_value_enum(self, input_value):
        """Construction normalizes underscores to match dash-valued enum."""
        result = DashValueEnum(input_value)
        assert result == DashValueEnum.MY_VALUE

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR"],
    )  # fmt: skip
    def test_eq_normalizes_dashes(self, input_value):
        """__eq__ normalizes dashes/underscores."""
        assert input_value == SampleEnum.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["my_value", "my-value", "MY_VALUE", "MY-VALUE"],
    )  # fmt: skip
    def test_eq_with_dash_value_enum(self, input_value):
        """__eq__ normalizes for dash-valued enums."""
        assert input_value == DashValueEnum.MY_VALUE

    def test_hash_normalized(self):
        """Hash is based on normalized value."""

        class EnumWithUnderscore(CaseInsensitiveStrEnum):
            ITEM = "foo_bar"

        class EnumWithDash(CaseInsensitiveStrEnum):
            ITEM = "foo-bar"

        # Same normalized value across different enums have same hash
        assert hash(EnumWithUnderscore.ITEM) == hash(EnumWithDash.ITEM)

    def test_hashable_in_collections(self):
        """Enum members work in sets and as dict keys."""
        enum_set = {SampleEnum.ALPHA, SampleEnum.BETA}
        assert len(enum_set) == 2
        assert SampleEnum.ALPHA in enum_set

        enum_dict = {SampleEnum.FOO_BAR: "value"}
        assert enum_dict[SampleEnum.FOO_BAR] == "value"


# =============================================================================
# Pydantic Integration Tests
# =============================================================================


class TestPydanticIntegration:
    """Tests for Pydantic model integration."""

    def test_enum_in_model(self):
        """Enum works as Pydantic model field type."""

        class Config(BaseModel):
            mode: SampleEnum

        config = Config(mode=SampleEnum.ALPHA)
        assert config.mode == SampleEnum.ALPHA

    def test_string_coercion(self):
        """Pydantic coerces string to enum."""

        class Config(BaseModel):
            mode: SampleEnum

        config = Config(mode="alpha")
        assert config.mode == SampleEnum.ALPHA

    def test_case_insensitive_coercion(self):
        """Pydantic coerces case-insensitive strings."""

        class Config(BaseModel):
            mode: SampleEnum

        config = Config(mode="ALPHA")
        assert config.mode == SampleEnum.ALPHA

    def test_dash_underscore_coercion(self):
        """Pydantic coerces dashed strings to underscore-valued enum."""

        class Config(BaseModel):
            mode: SampleEnum

        config = Config(mode="foo-bar")
        assert config.mode == SampleEnum.FOO_BAR

    def test_underscore_to_dash_coercion(self):
        """Pydantic coerces underscored strings to dash-valued enum."""

        class Config(BaseModel):
            mode: DashValueEnum

        config = Config(mode="my_value")
        assert config.mode == DashValueEnum.MY_VALUE

    def test_invalid_value_validation_error(self):
        """Pydantic raises ValidationError for invalid values."""

        class Config(BaseModel):
            mode: SampleEnum

        with pytest.raises(ValidationError):
            Config(mode="invalid")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and potential gotchas."""

    def test_enum_with_same_normalized_values(self):
        """Enums with different values but same normalized form compare equal."""

        class EnumA(CaseInsensitiveStrEnum):
            ITEM = "foo_bar"

        class EnumB(CaseInsensitiveStrEnum):
            ITEM = "foo-bar"

        assert EnumA.ITEM == EnumB.ITEM
        assert EnumA.ITEM == "foo-bar"
        assert EnumA.ITEM == "foo_bar"
        assert EnumB.ITEM == "foo-bar"
        assert EnumB.ITEM == "foo_bar"

    def test_invalid_value_raises_valueerror(self):
        """Invalid values raise ValueError on construction."""
        with pytest.raises(ValueError):
            SampleEnum("nonexistent")

    def test_multiple_dashes_normalized(self):
        """Multiple dashes are normalized to multiple underscores."""

        class MultiDashEnum(CaseInsensitiveStrEnum):
            MULTI = "foo--bar"

        assert MultiDashEnum("foo__bar") == MultiDashEnum.MULTI
        assert MultiDashEnum.MULTI == "foo__bar"

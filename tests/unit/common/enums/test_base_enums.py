# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CaseInsensitiveStrEnum and dash/underscore normalization."""

from typing import Self

import pytest
from pydantic import BaseModel, ValidationError
from pytest import param

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


# =============================================================================
# Normalization Caching Tests
# =============================================================================


class TestNormalizationCaching:
    """Tests for the per-member cached normalized value/hash hot-path."""

    def test_norm_value_cached_on_construction(self: Self) -> None:
        """Members precompute the normalized value once during construction."""
        assert SampleEnum.FOO_BAR.__dict__.get("_norm_value_cache") == "foo_bar"
        assert DashValueEnum.MY_VALUE.__dict__.get("_norm_value_cache") == "my_value"

    def test_norm_hash_cached_on_construction(self: Self) -> None:
        """Members precompute the normalized hash once during construction."""
        member = DashValueEnum.MY_VALUE
        assert member.__dict__.get("_norm_hash_cache") == hash("my_value")
        assert hash(member) == hash("my_value")

    def test_exact_match_fast_path(self: Self) -> None:
        """An exact string match compares equal without normalization work."""
        assert SampleEnum.FOO_BAR == "foo_bar"
        # Non-exact but normalized-equal still matches via the fallback.
        assert SampleEnum.FOO_BAR == "FOO-BAR"

    def test_identity_fast_path(self: Self) -> None:
        """A member is equal to itself via the identity short-circuit."""
        member = SampleEnum.ALPHA
        assert member == member

    def test_lazy_norm_value_fallback(self: Self) -> None:
        """A member missing the cached attr recomputes and caches lazily.

        A non-exact (normalized-equal) compare misses the exact-match fast path
        and forces the normalization fallback, which repopulates the cache.
        """
        member = SampleEnum.BETA
        # Simulate a member that skipped __init__ (e.g. dynamic creation).
        member.__dict__.pop("_norm_value_cache", None)
        member.__dict__.pop("_norm_hash_cache", None)
        assert member == "BETA"  # non-exact -> normalization path
        assert member.__dict__.get("_norm_value_cache") == "beta"
        assert hash(member) == hash("beta")
        assert member.__dict__.get("_norm_hash_cache") == hash("beta")

    def test_enum_vs_enum_uses_cached_norm(self: Self) -> None:
        """Cross-enum equality still holds using cached normalized values."""

        class EnumA(CaseInsensitiveStrEnum):
            ITEM = "foo_bar"

        class EnumB(CaseInsensitiveStrEnum):
            ITEM = "foo-bar"

        assert EnumA.ITEM == EnumB.ITEM
        assert hash(EnumA.ITEM) == hash(EnumB.ITEM)

    def test_non_str_enum_not_equal(self: Self) -> None:
        """An Enum whose value is not a str does not compare equal."""
        from enum import Enum

        class IntEnum(Enum):
            X = 1

        assert SampleEnum.ALPHA != IntEnum.X


# =============================================================================
# __ne__ / __eq__ Symmetry Tests
# =============================================================================


class TestNotEqualMirrorsEqual:
    """`!=` must be the exact negation of `==` for every operand shape.

    Regression guard: `str.__ne__` sits between these classes and `object` in
    the MRO, so an inherited `__ne__` silently bypasses the normalizing
    `__eq__` and makes `a == b` and `a != b` both True.
    """

    @pytest.mark.parametrize(
        "other",
        [
            "alpha",
            "ALPHA",
            "Alpha",
            "aLpHa",
            "beta",
            "nonexistent",
            123,
            None,
            [],
            SampleEnum.ALPHA,
            SampleEnum.BETA,
        ],
    )  # fmt: skip
    def test_ne_is_negation_of_eq(self: Self, other: object) -> None:
        """`member != other` equals `not (member == other)` for any operand."""
        assert (SampleEnum.ALPHA != other) is not (SampleEnum.ALPHA == other)  # noqa: SIM300 - enum must stay on the left; that is the operand order under test

    @pytest.mark.parametrize(
        "other",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo-Bar", "baz"],
    )  # fmt: skip
    def test_ne_is_negation_of_eq_across_dash_forms(self: Self, other: object) -> None:
        """Dash/underscore/case variants negate consistently."""
        assert (SampleEnum.FOO_BAR != other) is not (SampleEnum.FOO_BAR == other)  # noqa: SIM300 - enum must stay on the left; that is the operand order under test

    @pytest.mark.parametrize(
        "other",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo-Bar"],
    )  # fmt: skip
    def test_ne_false_for_normalized_match(self: Self, other: str) -> None:
        """A normalized match is never `!=`."""
        assert (SampleEnum.FOO_BAR != other) is False  # noqa: SIM300 - enum must stay on the left; that is the operand order under test

    @pytest.mark.parametrize(
        "other",
        ["my_value", "my-value", "MY_VALUE", "MY-VALUE"],
    )  # fmt: skip
    def test_ne_false_for_dash_valued_enum(self: Self, other: str) -> None:
        """Dash-valued members are not `!=` their underscore spellings."""
        assert (DashValueEnum.MY_VALUE != other) is False  # noqa: SIM300 - enum must stay on the left; that is the operand order under test

    @pytest.mark.parametrize(
        "other",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo-Bar"],
    )  # fmt: skip
    def test_ne_false_with_string_on_the_left(self: Self, other: str) -> None:
        """Reflected `!=` also routes through the normalizing comparison."""
        assert (other != SampleEnum.FOO_BAR) is False

    def test_ne_true_for_different_member(self: Self) -> None:
        """Genuinely different members stay `!=`."""
        assert (SampleEnum.ALPHA != SampleEnum.BETA) is True

    def test_ne_false_across_enums_with_same_normalized_value(self: Self) -> None:
        """Cross-enum members sharing a normalized value are not `!=`."""

        class EnumA(CaseInsensitiveStrEnum):
            ITEM = "foo_bar"

        class EnumB(CaseInsensitiveStrEnum):
            ITEM = "foo-bar"

        assert (EnumA.ITEM != EnumB.ITEM) is False

    def test_ne_true_for_non_string_types(self: Self) -> None:
        """Non-string, non-enum operands remain `!=`."""
        assert (SampleEnum.ALPHA != 123) is True
        assert (SampleEnum.ALPHA != None) is True  # noqa: E711 - testing __ne__ with None
        assert (SampleEnum.ALPHA != []) is True

    @pytest.mark.parametrize(
        "other",
        [
            param(123, id="int"),
            param(None, id="none"),
            param([], id="list"),
            param(4.5, id="float"),
        ],
    )  # fmt: skip
    def test_ne_forwards_notimplemented_for_unsupported_operands(
        self: Self, other: object
    ) -> None:
        """__ne__ defers to the other operand instead of asserting inequality.

        The operator-level check above cannot see this: `!= 123` is True either
        way, whether __ne__ forwards NotImplemented or wrongly hardcodes True.
        Only the direct call distinguishes them, and the difference matters for
        any third-party type whose reflected __eq__ should decide the result.
        """
        assert SampleEnum.ALPHA.__ne__(other) is NotImplemented

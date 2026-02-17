# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ExtensibleStrEnum and related utilities."""

import pytest
from pydantic import BaseModel, ValidationError

from aiperf.plugin.extensible_enums import (
    ExtensibleStrEnum,
    _normalize_name,
    create_enum,
)

# =============================================================================
# Fixtures
# =============================================================================


class SampleEnum(ExtensibleStrEnum):
    """Immutable sample enum for read-only tests."""

    ALPHA = "alpha"
    BETA = "beta"


@pytest.fixture
def fresh_enum():
    """Create a fresh enum class for tests that mutate state."""

    class TestEnum(ExtensibleStrEnum):
        BASE = "base"

    return TestEnum


@pytest.fixture
def enum_with_extension(fresh_enum):
    """Create enum with one registered extension."""
    fresh_enum.register("EXT", "ext")
    return fresh_enum


# =============================================================================
# Metaclass Tests
# =============================================================================


class TestExtensibleStrEnumMeta:
    """Tests for ExtensibleStrEnumMeta metaclass behavior."""

    def test_extensions_dict_created(self):
        """Enum classes have _extensions dict on creation."""
        assert hasattr(SampleEnum, "_extensions")
        assert isinstance(SampleEnum._extensions, dict)

    def test_getattr_extension_member(self, enum_with_extension):
        """Attribute access works for registered extensions."""
        assert enum_with_extension.EXT.value == "ext"
        assert enum_with_extension.EXT.name == "EXT"

    def test_getattr_missing_raises(self, fresh_enum):
        """AttributeError raised for non-existent attributes."""
        with pytest.raises(AttributeError, match="has no attribute 'MISSING'"):
            _ = fresh_enum.MISSING

    def test_dir_includes_extensions(self, fresh_enum):
        """dir() includes base and extension members."""
        fresh_enum.register("EXT_ONE", "ext_one")
        fresh_enum.register("EXT_TWO", "ext_two")

        dir_result = dir(fresh_enum)
        assert "BASE" in dir_result
        assert "EXT_ONE" in dir_result
        assert "EXT_TWO" in dir_result

    @pytest.mark.parametrize(
        "item,expected",
        [
            ("alpha", True),
            ("ALPHA", True),
            ("Alpha", True),
            ("beta", True),
            ("nonexistent", False),
            (123, False),
            (None, False),
        ],
    )  # fmt: skip
    def test_contains(self, item, expected):
        """__contains__ handles strings (case-insensitive) and non-strings."""
        assert (item in SampleEnum) == expected

    def test_contains_enum_member(self):
        """__contains__ works for enum member objects."""
        assert SampleEnum.ALPHA in SampleEnum
        assert SampleEnum.BETA in SampleEnum

    def test_contains_extension(self, enum_with_extension):
        """__contains__ works for extension members."""
        assert "ext" in enum_with_extension
        assert "EXT" in enum_with_extension
        assert enum_with_extension.EXT in enum_with_extension

    def test_iter_all_members(self, enum_with_extension):
        """Iteration yields base and extension members."""
        members = list(enum_with_extension)
        assert len(members) == 2
        values = [m.value for m in members]
        assert "base" in values
        assert "ext" in values

    def test_getitem_base_and_extension(self, enum_with_extension):
        """__getitem__ retrieves both base and extension members."""
        assert enum_with_extension["BASE"].value == "base"
        assert enum_with_extension["EXT"].value == "ext"

    def test_getitem_missing_raises(self):
        """KeyError raised for non-existent member names."""
        with pytest.raises(KeyError, match="is not a valid SampleEnum member"):
            _ = SampleEnum["NONEXISTENT"]

    def test_len_with_extensions(self, fresh_enum):
        """__len__ counts base + extension members."""
        assert len(fresh_enum) == 1
        fresh_enum.register("EXT1", "ext1")
        assert len(fresh_enum) == 2
        fresh_enum.register("EXT2", "ext2")
        assert len(fresh_enum) == 3


# =============================================================================
# Enum Instance Tests
# =============================================================================


class TestExtensibleStrEnum:
    """Tests for ExtensibleStrEnum instance behavior."""

    @pytest.mark.parametrize(
        "member,expected_str,expected_repr,expected_name,expected_value",
        [
            (SampleEnum.ALPHA, "alpha", "SampleEnum.ALPHA", "ALPHA", "alpha"),
            (SampleEnum.BETA, "beta", "SampleEnum.BETA", "BETA", "beta"),
        ],
    )  # fmt: skip
    def test_str_repr_name_value(
        self, member, expected_str, expected_repr, expected_name, expected_value
    ):
        """str/repr/name/value return expected values."""
        assert str(member) == expected_str
        assert repr(member) == expected_repr
        assert member.name == expected_name
        assert member.value == expected_value

    def test_extension_str_repr_name_value(self, enum_with_extension):
        """Extension members have correct str/repr/name/value."""
        ext = enum_with_extension.EXT
        assert str(ext) == "ext"
        assert repr(ext) == "TestEnum.EXT"
        assert ext.name == "EXT"
        assert ext.value == "ext"

    @pytest.mark.parametrize(
        "compare_value,expected",
        [
            ("alpha", True),
            ("ALPHA", True),
            ("Alpha", True),
            ("aLpHa", True),
            ("beta", False),
            (123, False),
            (None, False),
            ([], False),
        ],
    )  # fmt: skip
    def test_eq_case_insensitive(self, compare_value, expected):
        """__eq__ is case-insensitive for strings, False for non-strings."""
        result = compare_value == SampleEnum.ALPHA
        assert result is expected

    def test_eq_same_value_different_enum(self):
        """Enums with same value compare equal."""

        class OtherEnum(ExtensibleStrEnum):
            ALPHA = "alpha"

        assert SampleEnum.ALPHA == OtherEnum.ALPHA

    def test_eq_enum_members(self):
        """Same member equals itself, different members don't."""
        assert SampleEnum.ALPHA == SampleEnum.ALPHA
        assert SampleEnum.ALPHA != SampleEnum.BETA

    def test_hash_case_insensitive(self):
        """Hash is based on lowercase value."""
        assert hash(SampleEnum.ALPHA) == hash("alpha")

    def test_hashable_in_collections(self):
        """Enum members work in sets and as dict keys."""
        enum_set = {SampleEnum.ALPHA, SampleEnum.BETA}
        assert len(enum_set) == 2
        assert SampleEnum.ALPHA in enum_set

        enum_dict = {SampleEnum.ALPHA: 1, SampleEnum.BETA: 2}
        assert enum_dict[SampleEnum.ALPHA] == 1

    def test_extension_hashable(self, enum_with_extension):
        """Extension members are hashable and work in collections."""
        assert hash(enum_with_extension.EXT) == hash("ext")

        enum_set = {enum_with_extension.BASE, enum_with_extension.EXT}
        assert len(enum_set) == 2

        d = {enum_with_extension.EXT: "value"}
        assert d[enum_with_extension.EXT] == "value"

    def test_str_subclass(self):
        """Enum members are str subclass (required for Pydantic)."""
        assert isinstance(SampleEnum.ALPHA, str)
        assert SampleEnum.ALPHA.upper() == "ALPHA"
        assert SampleEnum.ALPHA.startswith("al")


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegister:
    """Tests for register() classmethod."""

    def test_register_returns_member(self, fresh_enum):
        """register() returns the created member."""
        result = fresh_enum.register("CUSTOM", "custom")
        assert result.name == "CUSTOM"
        assert result.value == "custom"

    def test_register_accessible_via_attribute(self, fresh_enum):
        """Registered member accessible as class attribute."""
        fresh_enum.register("CUSTOM", "custom")
        assert fresh_enum.CUSTOM.value == "custom"

    def test_register_appears_in_iteration(self, fresh_enum):
        """Registered member appears in iteration."""
        fresh_enum.register("CUSTOM", "custom")
        values = [m.value for m in fresh_enum]
        assert "custom" in values

    def test_register_duplicate_base_raises(self, fresh_enum):
        """Registering existing base member name raises ValueError."""
        with pytest.raises(ValueError, match="already defined in TestEnum"):
            fresh_enum.register("BASE", "different")

    def test_register_duplicate_extension_raises(self, fresh_enum):
        """Registering same extension twice raises ValueError."""
        fresh_enum.register("CUSTOM", "custom")
        with pytest.raises(ValueError, match="already registered as an extension"):
            fresh_enum.register("CUSTOM", "different")

    def test_register_multiple(self, fresh_enum):
        """Multiple extensions can be registered."""
        fresh_enum.register("EXT1", "ext1")
        fresh_enum.register("EXT2", "ext2")
        fresh_enum.register("EXT3", "ext3")

        assert len(fresh_enum) == 4
        assert fresh_enum.EXT1.value == "ext1"
        assert fresh_enum.EXT2.value == "ext2"
        assert fresh_enum.EXT3.value == "ext3"


# =============================================================================
# values() and names() Tests
# =============================================================================


class TestValuesAndNames:
    """Tests for values() and names() class methods."""

    def test_values_base(self):
        """values() returns base member values."""
        values = SampleEnum.values()
        assert isinstance(values, list)
        assert "alpha" in values
        assert "beta" in values

    def test_values_with_extensions(self, enum_with_extension):
        """values() includes extension values."""
        values = enum_with_extension.values()
        assert "base" in values
        assert "ext" in values

    def test_names_base(self):
        """names() returns base member names."""
        names = SampleEnum.names()
        assert isinstance(names, list)
        assert "ALPHA" in names
        assert "BETA" in names

    def test_names_with_extensions(self, enum_with_extension):
        """names() includes extension names."""
        names = enum_with_extension.names()
        assert "BASE" in names
        assert "EXT" in names


# =============================================================================
# Case-Insensitive Lookup Tests
# =============================================================================


class TestCaseInsensitiveLookup:
    """Tests for _missing_() case-insensitive lookup."""

    @pytest.mark.parametrize(
        "input_value,expected_name",
        [
            ("alpha", "ALPHA"),
            ("ALPHA", "ALPHA"),
            ("Alpha", "ALPHA"),
            ("aLpHa", "ALPHA"),
            ("beta", "BETA"),
            ("BETA", "BETA"),
        ],
    )  # fmt: skip
    def test_case_insensitive_construction(self, input_value, expected_name):
        """Enum construction is case-insensitive."""
        result = SampleEnum(input_value)
        assert result.name == expected_name

    def test_extension_lookup(self, enum_with_extension):
        """Case-insensitive lookup works for extensions."""
        assert enum_with_extension("ext").value == "ext"
        assert enum_with_extension("EXT").value == "ext"
        assert enum_with_extension("Ext").value == "ext"

    def test_invalid_value_raises(self):
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError):
            SampleEnum("nonexistent")


# =============================================================================
# create_enum() Tests
# =============================================================================


class TestCreateEnum:
    """Tests for create_enum() factory function."""

    def test_basic_creation(self):
        """create_enum creates enum from dict."""
        MyEnum = create_enum("MyEnum", {"ONE": "one", "TWO": "two"}, module=__name__)
        assert MyEnum.ONE.value == "one"
        assert MyEnum.TWO.value == "two"

    def test_extensible(self):
        """Created enum supports registration."""
        MyEnum = create_enum("MyEnum", {"BASE": "base"}, module=__name__)
        MyEnum.register("EXT", "ext")
        assert MyEnum.EXT.value == "ext"

    def test_case_insensitive(self):
        """Created enum supports case-insensitive lookup."""
        MyEnum = create_enum("MyEnum", {"ITEM": "item"}, module=__name__)
        assert MyEnum("item") == MyEnum.ITEM
        assert MyEnum("ITEM") == MyEnum.ITEM
        assert MyEnum("Item") == MyEnum.ITEM

    def test_str_subclass(self):
        """Created enum members are str subclass."""
        MyEnum = create_enum("MyEnum", {"ITEM": "item"}, module=__name__)
        assert isinstance(MyEnum.ITEM, str)

    def test_iteration_and_len(self):
        """Created enum supports iteration and len."""
        MyEnum = create_enum("MyEnum", {"A": "a", "B": "b", "C": "c"}, module=__name__)
        assert len(MyEnum) == 3
        values = sorted(m.value for m in MyEnum)
        assert values == ["a", "b", "c"]

    def test_module_required(self):
        """create_enum requires module parameter."""
        with pytest.raises(TypeError):
            create_enum("MyEnum", {"ITEM": "item"})  # type: ignore[call-arg]

    def test_module_explicit(self):
        """create_enum sets module from explicit argument."""
        MyEnum = create_enum("MyEnum", {"ITEM": "item"}, module="custom.module")
        assert MyEnum.__module__ == "custom.module"

    def test_empty_members(self):
        """create_enum handles empty dict."""
        MyEnum = create_enum("EmptyEnum", {}, module=__name__)
        assert len(MyEnum) == 0
        assert list(MyEnum) == []


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

    def test_invalid_value_validation_error(self):
        """Pydantic raises ValidationError for invalid values."""

        class Config(BaseModel):
            mode: SampleEnum

        with pytest.raises(ValidationError):
            Config(mode="invalid")

    def test_dynamic_enum_in_model(self):
        """Dynamically created enum works in Pydantic model."""
        DynamicEnum = create_enum("DynamicEnum", {"A": "a", "B": "b"}, module=__name__)

        class Config(BaseModel):
            value: DynamicEnum

        config = Config(value="a")
        assert config.value == DynamicEnum.A


# =============================================================================
# Isolation Tests
# =============================================================================


class TestIsolation:
    """Tests verifying enum class isolation."""

    def test_fresh_enum_no_extensions(self, fresh_enum):
        """Newly created enum has no extensions."""
        assert len(fresh_enum._extensions) == 0

    def test_extensions_not_shared(self):
        """Extensions don't leak between enum classes."""

        class Enum1(ExtensibleStrEnum):
            A = "a"

        class Enum2(ExtensibleStrEnum):
            B = "b"

        Enum1.register("SHARED", "value1")

        assert "SHARED" in Enum1._extensions
        assert "SHARED" not in Enum2._extensions

    def test_base_members_unchanged(self, fresh_enum):
        """Base members unchanged after extension registration."""
        original = list(fresh_enum.__members__.keys())
        fresh_enum.register("EXT", "ext")
        after = list(fresh_enum.__members__.keys())

        assert original == after
        assert "EXT" not in fresh_enum.__members__
        assert "EXT" in fresh_enum._extensions

    def test_independent_extensions(self):
        """Same extension name can exist in different enums."""

        class EnumA(ExtensibleStrEnum):
            A = "a"

        class EnumB(ExtensibleStrEnum):
            B = "b"

        EnumA.register("EXT", "ext_a")
        EnumB.register("EXT", "ext_b")

        assert EnumA.EXT.value == "ext_a"
        assert EnumB.EXT.value == "ext_b"


# =============================================================================
# Dash/Underscore Normalization Tests
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


class TestDashUnderscoreNormalization:
    """Tests for dash/underscore normalization in enum operations."""

    @pytest.fixture
    def enum_with_underscores(self):
        """Enum with underscore values (common Python convention)."""

        class UnderscoreEnum(ExtensibleStrEnum):
            FOO_BAR = "foo_bar"
            BAZ_QUX = "baz_qux"

        return UnderscoreEnum

    @pytest.fixture
    def enum_with_dashes(self):
        """Enum with dash values (common CLI convention)."""

        class DashEnum(ExtensibleStrEnum):
            FOO_BAR = "foo-bar"
            BAZ_QUX = "baz-qux"

        return DashEnum

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo_Bar", "Foo-Bar"],
    )  # fmt: skip
    def test_construction_with_underscore_value(
        self, enum_with_underscores, input_value
    ):
        """Construction normalizes dashes/underscores and case."""
        result = enum_with_underscores(input_value)
        assert result == enum_with_underscores.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR", "Foo_Bar", "Foo-Bar"],
    )  # fmt: skip
    def test_construction_with_dash_value(self, enum_with_dashes, input_value):
        """Construction normalizes dashes/underscores and case for dash-valued enums."""
        result = enum_with_dashes(input_value)
        assert result == enum_with_dashes.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR"],
    )  # fmt: skip
    def test_eq_with_underscore_value(self, enum_with_underscores, input_value):
        """__eq__ normalizes dashes/underscores and case."""
        assert input_value == enum_with_underscores.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR"],
    )  # fmt: skip
    def test_eq_with_dash_value(self, enum_with_dashes, input_value):
        """__eq__ normalizes dashes/underscores for dash-valued enums."""
        assert input_value == enum_with_dashes.FOO_BAR

    @pytest.mark.parametrize(
        "input_value",
        ["foo_bar", "foo-bar", "FOO_BAR", "FOO-BAR"],
    )  # fmt: skip
    def test_contains_normalized(self, enum_with_underscores, input_value):
        """__contains__ normalizes dashes/underscores and case."""
        assert input_value in enum_with_underscores

    def test_hash_normalized(self):
        """Hash is based on normalized value (dashes become underscores)."""

        class EnumA(ExtensibleStrEnum):
            FOO_BAR = "foo_bar"

        class EnumB(ExtensibleStrEnum):
            FOO_BAR = "foo-bar"

        # Same normalized value across different enums have same hash
        assert hash(EnumA.FOO_BAR) == hash(EnumB.FOO_BAR)

    def test_extension_dash_underscore(self):
        """Extensions work with dash/underscore normalization."""

        class TestEnum(ExtensibleStrEnum):
            BASE = "base"

        TestEnum.register("FOO_BAR", "foo_bar")

        # Can look up with dashes
        assert TestEnum("foo-bar") == TestEnum.FOO_BAR
        assert "foo-bar" in TestEnum
        assert TestEnum.FOO_BAR == "foo-bar"

    def test_pydantic_coerces_dashes(self):
        """Pydantic coerces dashed strings to underscore-valued enum."""

        class TestEnum(ExtensibleStrEnum):
            MY_VALUE = "my_value"

        class Config(BaseModel):
            field: TestEnum

        # Pydantic should coerce "my-value" to MY_VALUE
        config = Config(field="my-value")
        assert config.field == TestEnum.MY_VALUE

    def test_pydantic_coerces_underscores_to_dashed_enum(self):
        """Pydantic coerces underscored strings to dash-valued enum."""

        class TestEnum(ExtensibleStrEnum):
            MY_VALUE = "my-value"  # Value has dashes

        class Config(BaseModel):
            field: TestEnum

        # Pydantic should coerce "my_value" to MY_VALUE
        config = Config(field="my_value")
        assert config.field == TestEnum.MY_VALUE

    def test_different_values_same_normalized_form(self):
        """Enums with different values but same normalized form compare equal."""

        class EnumA(ExtensibleStrEnum):
            ITEM = "foo_bar"

        class EnumB(ExtensibleStrEnum):
            ITEM = "foo-bar"

        # These should compare equal because their normalized values match
        assert EnumA.ITEM == EnumB.ITEM
        assert EnumA.ITEM == "foo-bar"
        assert EnumA.ITEM == "foo_bar"
        assert EnumB.ITEM == "foo-bar"
        assert EnumB.ITEM == "foo_bar"

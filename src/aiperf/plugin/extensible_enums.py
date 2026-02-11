# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from enum import Enum

from typing_extensions import Self


def _normalize_name(value: str) -> str:
    """Normalize a string for comparison: lowercase and convert dashes to underscores.

    This enables flexible matching where 'foo-bar', 'foo_bar', 'FOO_BAR', and 'FOO-BAR'
    all match each other when used for enum lookups or plugin name resolution.
    """
    return value.lower().replace("-", "_")


class ExtensibleStrEnumMeta(type(Enum)):
    """Metaclass for extensible enums that support runtime registration.

    This metaclass enables enums to be extended with new members after class creation,
    which is useful for plugin systems where types are discovered at runtime.

    Features:
        - Dynamic member registration via _extensions dict
        - Case-insensitive lookups via _missing_
        - Iteration includes both base and extension members
        - __contains__ works for both names and values
    """

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls._extensions: dict[str, Self] = {}  # type: ignore
        return cls

    def __getattr__(cls, name: str) -> Self:
        """Allow attribute access to dynamically registered enum members."""
        if hasattr(cls, "_extensions") and name in cls._extensions:
            return cls._extensions[name]  # type: ignore
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __dir__(cls) -> list[str]:
        """Include dynamically registered members in dir() for IDE support."""
        return list(super().__dir__()) + list(getattr(cls, "_extensions", {}).keys())

    def __contains__(cls, item: object) -> bool:
        """Check membership for both base members and extensions."""
        if isinstance(item, str):
            normalized_item = _normalize_name(item)
            for member in cls.__members__.values():
                if _normalize_name(member.value) == normalized_item:
                    return True
            for ext_member in cls._extensions.values():
                if _normalize_name(ext_member.value) == normalized_item:
                    return True
            return False
        return item in cls.__members__.values() or item in cls._extensions.values()

    def __iter__(cls) -> Iterator[Self]:
        """Iterate over all enum members including extensions."""
        yield from cls.__members__.values()
        yield from cls._extensions.values()

    def __getitem__(cls, item: str) -> Self:
        """Get enum member by name (base or extension)."""
        if item in cls.__members__:
            return cls.__members__[item]
        if item in cls._extensions:
            return cls._extensions[item]
        raise KeyError(f"'{item}' is not a valid {cls.__name__} member")

    def __len__(cls) -> int:
        """Return total count of base members plus extensions."""
        return len(cls.__members__) + len(cls._extensions)


class ExtensibleStrEnum(str, Enum, metaclass=ExtensibleStrEnumMeta):
    """String enum that supports runtime extension for plugin systems.

    This enum class combines the benefits of Python's Enum with dynamic extensibility:
    - Works with Pydantic validation (it's a str subclass)
    - Works with cyclopts CLI (has __iter__ for choices)
    - Supports case-insensitive lookups
    - Can be extended at runtime via register()

    Usage:
        # Define base enum with known members
        class EndpointType(ExtensibleStrEnum):
            CHAT = "chat"
            COMPLETIONS = "completions"

        # Extend at runtime (e.g., from plugin discovery)
        EndpointType.register("CUSTOM", "custom")

        # Use normally
        endpoint = EndpointType.CHAT
        endpoint = EndpointType("chat")  # case-insensitive
        endpoint = EndpointType.CUSTOM   # works after registration

    For plugin registries, use create_enum() to generate the entire enum dynamically.
    """

    _extensions: dict[str, Self]

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return _normalize_name(self.value) == _normalize_name(other)
        if hasattr(other, "value") and isinstance(other.value, str):
            return _normalize_name(self.value) == _normalize_name(other.value)
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(_normalize_name(self.value))

    @property
    def name(self) -> str:
        if hasattr(self, "_name_"):
            return self._name_
        return super().name

    @property
    def value(self) -> str:
        if hasattr(self, "_value_"):
            return self._value_
        return super().value

    @classmethod
    def register(cls, name: str, value: str) -> Self:
        """Register a new enum member at runtime.

        Args:
            name: The member name (e.g., "CUSTOM_ENDPOINT")
            value: The string value (e.g., "custom_endpoint")

        Returns:
            The newly created enum member

        Raises:
            ValueError: If name already exists, or if value conflicts with existing member
        """
        if name in cls.__members__:
            raise ValueError(f"'{name}' is already defined in {cls.__name__}")
        if name in cls._extensions:
            raise ValueError(
                f"'{name}' is already registered as an extension in {cls.__name__}"
            )

        extension_member = cls._create_extension_member(name, value)
        cls._extensions[name] = extension_member
        return extension_member

    @classmethod
    def _create_extension_member(cls, name: str, value: str) -> Self:
        """Create an extension member that behaves like a real enum member."""
        obj = str.__new__(cls, value)
        obj._name_ = name
        obj._value_ = value
        obj.__class__ = cls
        return obj

    @classmethod
    def values(cls) -> list[str]:
        """Get all string values including extensions."""
        base_values = [member.value for member in cls.__members__.values()]
        extension_values = [member.value for member in cls._extensions.values()]
        return base_values + extension_values

    @classmethod
    def names(cls) -> list[str]:
        """Get all member names including extensions."""
        base_names = list(cls.__members__.keys())
        extension_names = list(cls._extensions.keys())
        return base_names + extension_names

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        """Handle case-insensitive and dash/underscore-normalized lookups for string values."""
        if isinstance(value, str):
            normalized_value = _normalize_name(value)
            for member in cls.__members__.values():
                if _normalize_name(member.value) == normalized_value:
                    return member
            for ext_member in cls._extensions.values():
                if _normalize_name(ext_member.value) == normalized_value:
                    return ext_member
        return None


def create_enum(
    name: str, members: dict[str, str], *, module: str
) -> type[ExtensibleStrEnum]:
    """Create a new ExtensibleStrEnum dynamically from a dict of members.

    This is the preferred way to create enums from plugin registries at runtime.
    The resulting enum has full Pydantic/cyclopts compatibility.

    Args:
        name: The enum class name (e.g., "EndpointType")
        members: Dict mapping member names to values (e.g., {"CHAT": "chat"})
        module: Module name for the enum. Required for pickling since pickle
            looks up classes by module.name.

    Returns:
        A new ExtensibleStrEnum subclass with the specified members

    Example:
        >>> EndpointType = create_enum("EndpointType", {"CHAT": "chat", "COMPLETIONS": "completions"}, module=__name__)
        >>> EndpointType.CHAT
        EndpointType.CHAT
        >>> EndpointType("chat")
        EndpointType.CHAT
    """
    enum_cls = ExtensibleStrEnum(name, members)  # type: ignore[return-value]

    # Set __module__ to enable pickling. Pickle looks up classes by module.name,
    # so the enum must be findable in its declared module.
    enum_cls.__module__ = module
    return enum_cls

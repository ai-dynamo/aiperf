# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from functools import cached_property
from typing import Self

from pydantic import BaseModel, Field


def _normalize_name(value: str) -> str:
    """Normalize a string for comparison: lowercase and convert dashes to underscores.

    This enables flexible matching where 'foo-bar', 'foo_bar', 'FOO_BAR', and 'FOO-BAR'
    all match each other when used for enum lookups or plugin name resolution.
    """
    return value.lower().replace("-", "_")


class CaseInsensitiveStrEnum(str, Enum):
    """
    CaseInsensitiveStrEnum is a custom enumeration class that extends `str` and `Enum` to provide case-insensitive
    lookup functionality for its members.
    """

    def __init__(self: Self, *args: object) -> None:
        # Comparisons and hashing sit on hot paths (per-SSE-chunk field checks
        # at >1M/s under load); normalize once per member instead of per call.
        self._norm_value_cache = _normalize_name(self.value)
        self._norm_hash_cache = hash(self._norm_value_cache)

    def _norm_value(self: Self) -> str:
        # Lazy fallback for members created outside Enum construction
        # (e.g. dynamically registered custom members), which skip __init__.
        norm = self.__dict__.get("_norm_value_cache")
        if norm is None:
            norm = _normalize_name(self.value)
            self._norm_value_cache = norm
        return norm

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __eq__(self: Self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, CaseInsensitiveStrEnum):
            return self._norm_value() == other._norm_value()
        if isinstance(other, str):
            # Exact match first: skips normalization for the common case where
            # the raw wire string already equals the member value.
            return str.__eq__(
                self, other
            ) is True or self._norm_value() == _normalize_name(other)
        if isinstance(other, Enum):
            return isinstance(other.value, str) and self._norm_value() == (
                _normalize_name(other.value)
            )
        return super().__eq__(other)

    def __hash__(self: Self) -> int:
        norm_hash = self.__dict__.get("_norm_hash_cache")
        if norm_hash is None:
            norm_hash = hash(self._norm_value())
            self._norm_hash_cache = norm_hash
        return norm_hash

    @classmethod
    def _missing_(cls, value):
        """
        Handles cases where a value is not directly found in the enumeration.

        This method is called when an attempt is made to access an enumeration
        member using a value that does not directly match any of the defined
        members. Supports case-insensitive matching and dash/underscore normalization.

        Returns:
            The matching enumeration member if a normalized match is found
            for string values; otherwise, returns None.
        """
        if isinstance(value, str):
            normalized_value = _normalize_name(value)
            for member in cls:
                if _normalize_name(member.value) == normalized_value:
                    return member
        return None


class BasePydanticEnumInfo(BaseModel):
    """Base class for all enum info classes that extend `BasePydanticBackedStrEnum`. By default, it
    provides a `tag` for the enum member, which is used for lookup and string comparison,
    and the subclass can provide additional information as needed."""

    tag: str = Field(
        ...,
        min_length=1,
        description="The string value of the enum member used for lookup, serialization, and string insensitive comparison.",
    )

    def __str__(self) -> str:
        return self.tag


class BasePydanticBackedStrEnum(CaseInsensitiveStrEnum):
    """
    Custom enumeration class that extends `CaseInsensitiveStrEnum`
    and is backed by a `BasePydanticEnumInfo` that contains the `tag`, and any other information that is needed
    to represent the enum member.
    """

    # Override the __new__ method to store the `BasePydanticEnumInfo` subclass model as an attribute. This is a python feature that
    # allows us to modify the behavior of the enum class's constructor. We use this to ensure the the enums still look like
    # a regular string enum, but also have the additional information stored as an attribute.
    def __new__(cls, info: BasePydanticEnumInfo) -> Self:
        # Create a new string object based on this class and the tag value.
        obj = str.__new__(cls, info.tag)
        # Ensure string value is set for comparison. This is how enums work internally.
        obj._value_ = info.tag
        # Store the Pydantic model as an attribute.
        obj._info: BasePydanticEnumInfo = info  # type: ignore
        return obj

    @cached_property
    def info(self) -> BasePydanticEnumInfo:
        """Get the enum info for the enum member."""
        # This is the Pydantic model that was stored as an attribute in the __new__ method.
        return self._info  # type: ignore

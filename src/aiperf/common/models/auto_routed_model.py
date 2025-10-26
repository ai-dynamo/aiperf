# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
High-performance auto-routed Pydantic model base class.

Faster than Pydantic's native discriminated unions through:
- Single JSON parse with zero-copy dict routing
- Automatic subclass registration via __init_subclass__
- Supports both cascading and hierarchical discriminators

Cascading discriminators (different field at each level):
    class Message(AutoRoutedModel):
        discriminator_field = "message_type"
        message_type: str

    class CommandMessage(Message):
        discriminator_field = "command"  # NEW discriminator field
        message_type: str = "command"
        command: str

    class SpawnWorkersCommand(CommandMessage):
        command: str = "spawn_workers"

Hierarchical discriminators (same field, multiple levels):
    class Animal(AutoRoutedModel):
        discriminator_field = "type"
        type: str

    class Dog(Animal):
        type: str = "dog"
        # discriminator_field inherited from Animal

    class Poodle(Dog):
        type: str = "poodle"
        # All register in Animal._model_lookup_table

    # Routes directly: Animal.from_json({"type": "poodle"}) -> Poodle instance
"""

from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing_extensions import Self

from aiperf.common.utils import load_json_str


class AutoRoutedModel(BaseModel):
    """High-performance model base with automatic multi-level routing."""

    discriminator_field: ClassVar[str | None] = None
    _model_lookup_table: ClassVar[dict[Any, type[Self]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        parent = cls.__bases__[0]
        parent_discriminator = getattr(parent, "discriminator_field", None)

        # Get the discriminator explicitly defined on THIS class (not inherited)
        cls_discriminator = cls.__dict__.get("discriminator_field", None)

        # If this class explicitly sets the SAME discriminator as parent, share lookup table
        # Otherwise, if it sets a NEW discriminator, create new lookup table
        if cls_discriminator == parent_discriminator and cls_discriminator is not None:
            # Same discriminator -> share parent's lookup table
            cls._model_lookup_table = getattr(parent, "_model_lookup_table", {})
        elif cls_discriminator is not None:
            # New discriminator -> create new lookup table
            cls._model_lookup_table = {}

        # Register this class in parent's lookup table if parent has a discriminator
        if parent_discriminator:
            discriminator_value = getattr(cls, parent_discriminator, None)

            # Unwrap Pydantic FieldInfo if needed
            if isinstance(discriminator_value, FieldInfo):
                discriminator_value = discriminator_value.default

            if discriminator_value is not None and hasattr(
                parent, "_model_lookup_table"
            ):
                parent._model_lookup_table[discriminator_value] = cls

    @classmethod
    def from_json(cls, json_or_dict: str | bytes | bytearray | dict) -> Self:
        """Single-parse JSON deserialization with zero-copy dict routing."""
        data = (
            json_or_dict
            if isinstance(json_or_dict, dict)
            else load_json_str(json_or_dict)
        )

        # Only route if THIS class explicitly set a discriminator (not inherited)
        cls_discriminator = cls.__dict__.get("discriminator_field")
        if cls_discriminator and cls._model_lookup_table:
            discriminator_value = data.get(cls_discriminator)

            if not discriminator_value:
                raise ValueError(
                    f"Missing discriminator '{cls_discriminator}' in: {data}"
                )

            target_class = cls._model_lookup_table.get(discriminator_value)
            if target_class:
                # Recurse for nested routing, otherwise validate
                return target_class.from_json(data)

        return cls.model_validate(data)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import msgspec
from pydantic import ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from aiperf.common.models.auto_routed_model import AutoRoutedModel


class AIPerfBaseModel(AutoRoutedModel):
    """Base model for all AIPerf Pydantic models.

    Inherits high-performance auto-routing capabilities from AutoRoutedModel.
    Models can optionally set discriminator_field to enable automatic routing.

    This class is configured to allow arbitrary types to be used as fields
    to allow for more flexible model definitions by end users without breaking
    existing code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PydanticStructMixin:
    """Shim letting Pydantic v2 parents accept/serialize a msgspec.Struct field.

    Pydantic cannot natively validate or JSON-serialize a msgspec.Struct. This
    teaches Pydantic to accept dicts/instances during validation (via
    ``msgspec.convert``) and to emit plain dicts during dump (via
    ``msgspec.to_builtins``), so wrappers like ``ConversationResponseMessage``
    continue to round-trip without special-casing at every callsite.

    Usage — mix into any msgspec.Struct that can appear as a field inside a
    Pydantic envelope::

        class MyPayload(
            PydanticStructMixin,
            msgspec.Struct,
            kw_only=True,
            omit_defaults=True,
        ):
            ...

    Part of the msgspec ZMQ migration (see
    docs/superpowers/specs/2026-04-20-msgspec-zmq-migration-overview.md).
    Generalized from the original ``_PydanticStructMixin`` introduced for
    dataset hot-path models in 073cc3011. Retired in the primitives spec
    (P2) when the Pydantic ``Message`` envelope itself becomes a
    msgspec.Struct and all field-level compat disappears.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def _validate(value: Any) -> Any:
            if isinstance(value, cls):
                return value
            # Pydantic resolves union members by trying each in order. If we
            # get a non-mapping (e.g. a `str` URL being validated against
            # `list[str] | list[Image]`), raise a plain ValueError so Pydantic
            # falls through to the next union variant rather than bubbling a
            # msgspec-specific error that older pydantic versions treat as
            # fatal.
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected dict or {cls.__name__} instance, got {type(value).__name__}"
                )
            return msgspec.convert(value, cls)

        def _serialize(value: Any) -> Any:
            return msgspec.to_builtins(value)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                return_schema=core_schema.any_schema(),
                when_used="always",
            ),
        )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, model_serializer

BaseModelT = TypeVar("BaseModelT", bound="AIPerfBaseModel")


class AIPerfBaseModel(BaseModel):
    """Base model for all AIPerf Pydantic models. This class is configured to allow
    arbitrary types to be used as fields as to allow for more flexible model definitions
    by end users without breaking the existing code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


def exclude_if_none(field_names: list[str]):
    """Decorator to set the _exclude_if_none_fields class attribute to the set of
    field names that should be excluded if they are None.
    """

    def decorator(model: type[BaseModelT]) -> type[BaseModelT]:
        if not hasattr(model, "_exclude_if_none_fields"):
            model._exclude_if_none_fields = set()
        model._exclude_if_none_fields.update(field_names)
        return model

    return decorator


class ExcludeIfNoneMixin(AIPerfBaseModel):
    """Mixin to exclude fields from the serialized model if they are None.

    The @exclude_if_none decorator can be used to specify which fields
    should be excluded from the serialized model if they are None.
    """

    _exclude_if_none_fields: ClassVar[set[str]] = set()
    """Set of field names that should be excluded from the serialized model if they
    are None. This is set by the @exclude_if_none decorator.
    """

    @model_serializer
    def _serialize_model(self) -> dict[str, Any]:
        """Serialize the model to a dictionary.

        This method overrides the default serializer to exclude fields that with a
        value of None and were marked with the @exclude_if_none decorator.
        """
        return {
            k: v
            for k, v in self
            if not (k in self._exclude_if_none_fields and v is None)
        }

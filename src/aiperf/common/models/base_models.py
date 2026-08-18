# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel, ConfigDict

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


def _msgspec_enc_hook(obj: Any) -> Any:
    """enc_hook for ``msgspec.to_builtins`` / ``msgspec.msgpack.Encoder``.

    Handles types that msgspec's built-in encoder does not recognise:
    - AIPerf's plugin-backed enums (``ExtensibleStrEnum``) use a custom
      metaclass so they fall through to ``isinstance(obj, Enum)``.
    - ``pathlib.PurePath`` / ``Path`` render to their string form.
    - numpy scalars (``float64``, ``int64``, ...) subclass float/int but
      msgspec's fast path keys on type identity, so coerce via ``.item()``
      to a builtin. Avoids an import of numpy in this module.
    - Pydantic ``BaseModel`` instances project through
      ``model_dump(mode='json')``.

    Everything else raises ``NotImplementedError`` and lets msgspec emit
    its standard unsupported-type error.

    Example:
        >>> from aiperf.common.enums import ModelSelectionStrategy
        >>> _msgspec_enc_hook(ModelSelectionStrategy.ROUND_ROBIN)
        'round_robin'
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise NotImplementedError(f"Objects of type {type(obj).__name__} are not supported")

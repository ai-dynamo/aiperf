# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel, ConfigDict


class AIPerfBaseModel(BaseModel):
    """Base model for all AIPerf Pydantic models.

    This class is configured to allow arbitrary types to be used as fields
    to allow for more flexible model definitions by end users without breaking
    existing code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


def _msgspec_enc_hook(obj: Any) -> Any:
    """enc_hook for msgspec.to_builtins.

    Handles types that msgspec's built-in encoder does not recognise:
    - AIPerf's plugin-backed enums (``ExtensibleStrEnum``) use a custom
      metaclass so they fall through to ``isinstance(obj, Enum)``.
    - ``pathlib.PurePath`` / ``Path`` render to their string form.
    - numpy scalars (``float64``, ``int64``, ...) subclass float/int but
      msgspec's fast path keys on type identity, so coerce via ``.item()``
      to a builtin. Avoids an import of numpy in this module.

    Everything else raises ``NotImplementedError`` and lets msgspec emit
    its standard unsupported-type error.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        return obj.item()
    raise NotImplementedError(f"Objects of type {type(obj).__name__} are not supported")


def _msgspec_dec_hook(target_type: type, obj: Any) -> Any:
    """dec_hook for msgspec.convert — symmetric to ``_msgspec_enc_hook``."""
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(obj)
    if isinstance(target_type, type) and issubclass(target_type, PurePath):
        return target_type(obj)
    raise NotImplementedError(f"Cannot decode {type(obj).__name__} as {target_type!r}")

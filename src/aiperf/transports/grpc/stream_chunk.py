# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol-agnostic streaming response chunk."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class StreamChunk:
    """Protocol-agnostic container for a single streaming response chunk.

    Decouples the transport layer from any specific protobuf schema by
    carrying pre-deserialized response data and error information.
    """

    error_message: str | None
    response_dict: dict[str, Any] | None
    response_size: int

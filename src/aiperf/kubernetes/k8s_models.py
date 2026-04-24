# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base model for Kubernetes resources with automatic camelCase serialization."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

from aiperf.common.models import AIPerfBaseModel


class K8sCamelModel(AIPerfBaseModel):
    """Base for Kubernetes resource models with camelCase serialization.

    Subclasses get automatic snake_case -> camelCase field aliasing via
    Pydantic's alias_generator. Use ``to_k8s_dict()`` or
    ``model_dump(by_alias=True)`` to serialize.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    def to_k8s_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase dict, excluding None values."""
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base configuration model with camelCase alias support.

All user-facing config models inherit BaseConfig so that:
- Serialization (model_dump / YAML / JSON) uses camelCase keys
- Deserialization accepts both camelCase and snake_case input
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseConfig(BaseModel):
    """Base for all AIPerf configuration models.

    Provides camelCase alias generation for K8s CRD compatibility
    while keeping Python field names as snake_case.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

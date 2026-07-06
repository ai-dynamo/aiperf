# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Weights & Biases - run-upload configuration.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig

__all__ = [
    "WandbConfig",
]


def _parse_wandb_tags(value: Any | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    tags = [str(tag).strip() for tag in value]
    return [tag for tag in tags if tag] or None


def _normalize_optional_str(value: Any | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


class WandbConfig(BaseConfig):
    """Weights & Biases run-upload configuration."""

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        # Mirror _require_project_for_secondary_fields in the generated JSON
        # schema so editors flag secondary keys authored without `project`.
        json_schema_extra={
            "dependentRequired": {
                "entity": ["project"],
                "runName": ["project"],
                "tags": ["project"],
            }
        },
    )

    project: Annotated[
        str | None,
        Field(
            default=None,
            description="Weights & Biases project name. Setting this enables wandb export.",
        ),
        BeforeValidator(_normalize_optional_str),
    ]
    entity: Annotated[
        str | None,
        Field(
            default=None,
            description="Weights & Biases entity (team or user). Defaults to the API key's default entity.",
        ),
        BeforeValidator(_normalize_optional_str),
    ]
    run_name: Annotated[
        str | None,
        Field(default=None, description="Weights & Biases run name."),
        BeforeValidator(_normalize_optional_str),
    ]
    tags: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Additional Weights & Biases run tags to attach on upload.",
        ),
        BeforeValidator(_parse_wandb_tags),
    ]

    @model_validator(mode="after")
    def _require_project_for_secondary_fields(self) -> WandbConfig:
        """Reject secondary wandb settings authored without a project, matching
        the CLI converter's contract for --wandb-entity/--wandb-run-name/--wandb-tag."""
        if self.project is None:
            set_secondary = [
                name
                for name in ("entity", "run_name", "tags")
                if getattr(self, name) is not None
            ]
            if set_secondary:
                raise ValueError(
                    f"wandb.{', wandb.'.join(set_secondary)} require "
                    "wandb.project to be set."
                )
        return self

    @property
    def enabled(self) -> bool:
        """Whether Weights & Biases export is enabled."""
        return self.project is not None

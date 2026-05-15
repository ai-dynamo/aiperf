# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

MLflow - Tracking and artifact-upload configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig
from aiperf.config.loader.parsing import parse_str_or_dict_as_tuple_list

__all__ = [
    "MLflowConfig",
    "MLflowDefaults",
]


@dataclass(frozen=True)
class MLflowDefaults:
    TRACKING_URI = None
    EXPERIMENT = "aiperf"
    RUN_NAME = None
    TAGS = None
    ARTIFACT_GLOBS = None
    DEFAULT_ARTIFACT_GLOBS = (
        "*.json",
        "*.csv",
        "*.jsonl",
        "*.parquet",
        "*_timeslices.*",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.svg",
        "**/*.html",
    )
    EXPORT_METADATA_FILE = Path("mlflow_export.json")


def _parse_mlflow_tags(value: Any | None) -> dict[str, str] | None:
    pairs = parse_str_or_dict_as_tuple_list(value)
    if pairs is None:
        return None
    tags: dict[str, str] = {}
    for key, tag_value in pairs:
        key_str = str(key).strip()
        if not key_str:
            continue
        tags[key_str] = str(tag_value)
    return tags


class MLflowConfig(BaseConfig):
    """MLflow tracking and artifact-upload configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    tracking_uri: Annotated[
        str | None,
        Field(default=MLflowDefaults.TRACKING_URI, description="MLflow tracking URI."),
    ]
    experiment: Annotated[
        str,
        Field(default=MLflowDefaults.EXPERIMENT, description="MLflow experiment name."),
    ]
    run_name: Annotated[
        str | None,
        Field(default=MLflowDefaults.RUN_NAME, description="MLflow run name."),
    ]
    tags: Annotated[
        dict[str, str] | None,
        Field(
            default=MLflowDefaults.TAGS,
            description="Additional MLflow run tags to attach on upload.",
        ),
        BeforeValidator(_parse_mlflow_tags),
    ]
    parent_run_id: Annotated[
        str | None,
        Field(default=None, description="Optional MLflow parent run ID."),
    ]
    artifact_globs: Annotated[
        list[str] | None,
        Field(
            default=MLflowDefaults.ARTIFACT_GLOBS,
            description="Artifact glob overrides for MLflow upload.",
        ),
    ]

    @property
    def enabled(self) -> bool:
        """Whether MLflow export/live streaming is enabled."""
        return self.tracking_uri is not None

    @property
    def tags_dict(self) -> dict[str, str]:
        """Get MLflow tags as a normalized dict[str, str]."""
        return dict(self.tags or {})

    @property
    def resolved_artifact_globs(self) -> list[str]:
        """Return MLflow artifact globs, applying defaults when unset."""
        return list(self.artifact_globs or MLflowDefaults.DEFAULT_ARTIFACT_GLOBS)

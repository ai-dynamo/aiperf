# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nested configs for endpoint control hooks (reset_kv_cache, server_profiler)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig

DEFAULT_RESET_KV_CACHE_PATH = "/reset_prefix_cache"
DEFAULT_SERVER_PROFILER_START_PATH = "/start_profile"
DEFAULT_SERVER_PROFILER_STOP_PATH = "/stop_profile"


class ResetKvCacheConfig(BaseConfig):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Timeout in seconds for the reset_kv_cache POST. "
            "When unset, uses endpoint.timeout.",
        ),
    ] = None
    path: Annotated[
        str | None,
        Field(
            default=None,
            description="Relative path for KV-cache reset. "
            f"Default: {DEFAULT_RESET_KV_CACHE_PATH}.",
        ),
    ] = None


class ServerProfilerConfig(BaseConfig):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            description="Timeout in seconds for profiler start/stop POSTs. "
            "When unset, uses endpoint.timeout.",
        ),
    ] = None
    start_path: Annotated[
        str | None,
        Field(
            default=None,
            description="Relative path for profiler start. "
            f"Default: {DEFAULT_SERVER_PROFILER_START_PATH}.",
        ),
    ] = None
    stop_path: Annotated[
        str | None,
        Field(
            default=None,
            description="Relative path for profiler stop. "
            f"Default: {DEFAULT_SERVER_PROFILER_STOP_PATH}.",
        ),
    ] = None


def require_relative_path(path: str | None, field_name: str) -> str | None:
    """Validate a control-hook path is relative and leading-slash, or None.

    Absolute URLs are rejected so hook paths always resolve against the
    endpoint's own origin rather than redirecting control traffic elsewhere.
    """
    if path is None:
        return None
    if "://" in path or not path.startswith("/"):
        raise ValueError(
            f"{field_name} must be a relative path starting with '/', got {path!r}"
        )
    return path


def parse_enabled_or_config(model_cls: type[BaseConfig]) -> BeforeValidator:
    """Build a BeforeValidator for ``false | true | object`` nested configs."""

    def _parse(value: Any) -> BaseConfig | None:
        if value is None or value is False:
            return None
        if value is True:
            return model_cls()
        if isinstance(value, model_cls):
            return value
        if isinstance(value, dict):
            return model_cls.model_validate(value)
        raise ValueError(
            f"Expected false | true | object for {model_cls.__name__}, got {type(value).__name__}"
        )

    return BeforeValidator(_parse)

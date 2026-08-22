# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator runtime settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OperatorSettings(BaseSettings):
    """Environment-derived settings for the isolated operator."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_OPERATOR_")

    index_rebuild_token: str = ""
    artifact_root: str = "/var/lib/aiperf/results"

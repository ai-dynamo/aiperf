# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator runtime settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OperatorSettings(BaseSettings):
    """Environment-derived settings for the isolated operator."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_K8S_OPERATOR_")

    artifact_root: str = "/var/lib/aiperf/results"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    results_upload_base_url: str = "http://aiperf-k8s-operator.aiperf-system.svc:8080"

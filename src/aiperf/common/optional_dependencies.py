# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for optional runtime dependency messaging."""

MLFLOW_EXTRA = "aiperf[mlflow]"
OTEL_EXTRA = "aiperf[otel]"
OTEL_METRICS_STREAMING_FEATURE = "OpenTelemetry metrics streaming is enabled"


def install_optional_dependency_hint(extra: str) -> str:
    return f"Install with: pip install '{extra}' or uv add '{extra}'."


def mlflow_dependency_message(feature: str) -> str:
    return (
        f"{feature} but the optional MLflow dependency is not installed. "
        f"{install_optional_dependency_hint(MLFLOW_EXTRA)}"
    )


def otel_dependency_message(feature: str) -> str:
    return (
        f"{feature} but the optional OpenTelemetry dependencies are not installed. "
        f"{install_optional_dependency_hint(OTEL_EXTRA)}"
    )

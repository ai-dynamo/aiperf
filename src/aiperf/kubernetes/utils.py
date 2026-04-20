# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes resource parsing and formatting utilities.

Consolidated functions for parsing Kubernetes CPU and memory strings
into numeric values, and formatting them back for display.
"""


def parse_cpu(value: str) -> float:
    """Parse Kubernetes CPU string to cores (float).

    Args:
        value: CPU string like '500m', '0.5', '2', '100m'.

    Returns:
        CPU value in cores (e.g., 0.5 for '500m', 2.0 for '2').
    """
    if not value or value == "0":
        return 0.0
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000_000
    if value.endswith("u"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("m"):
        return float(value[:-1]) / 1000
    return float(value)


def parse_memory_mib(value: str) -> int:
    """Parse Kubernetes memory string to MiB (int).

    Args:
        value: Memory string like '256Mi', '1Gi', '1024Ki'.

    Returns:
        Memory value in MiB.
    """
    if not value or value == "0":
        return 0
    if value.endswith("Gi"):
        return int(float(value[:-2]) * 1024)
    if value.endswith("Mi"):
        return int(float(value[:-2]))
    if value.endswith("Ki"):
        return max(1, int(float(value[:-2]) / 1024))
    return int(value)


def parse_memory_gib(value: str) -> float:
    """Parse Kubernetes memory string to GiB (float).

    Handles Gi, Mi, G, M, Ki suffixes and raw bytes.

    Args:
        value: Memory string like '1Gi', '512Mi', '1024M'.

    Returns:
        Memory in GiB.
    """
    value = value.strip()
    if not value or value == "0":
        return 0.0
    if value.endswith("Gi"):
        return float(value[:-2])
    if value.endswith("Mi"):
        return float(value[:-2]) / 1024
    if value.endswith("G"):
        return float(value[:-1]) * 1000 / 1024  # GB to GiB
    if value.endswith("M"):
        return float(value[:-1]) / 1024
    if value.endswith("Ki"):
        return float(value[:-2]) / 1024 / 1024
    return float(value) / 1024 / 1024 / 1024  # bytes to GiB


def format_cpu(cores: float) -> str:
    """Format CPU cores for display.

    Args:
        cores: CPU value in cores.

    Returns:
        Formatted string (e.g., '500m' for 0.5, '2.0' for 2.0).
    """
    if cores < 1:
        return f"{int(cores * 1000)}m"
    return f"{cores:.1f}"


def format_memory(gib: float) -> str:
    """Format memory GiB for display.

    Args:
        gib: Memory value in GiB.

    Returns:
        Formatted string (e.g., '512Mi' for 0.5, '2.0Gi' for 2.0).
    """
    if gib < 1:
        return f"{int(gib * 1024)}Mi"
    return f"{gib:.1f}Gi"

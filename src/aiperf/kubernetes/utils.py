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


# Kubernetes quantity suffixes. Binary (Ki/Mi/...) must be tried before the
# decimal ones, since "Mi" also ends with "i" but "M" is a prefix of "Mi".
_MEMORY_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("Ki", 1024),
    ("Mi", 1024**2),
    ("Gi", 1024**3),
    ("Ti", 1024**4),
    ("Pi", 1024**5),
    ("Ei", 1024**6),
    ("k", 1000),
    ("K", 1000),
    ("M", 1000**2),
    ("G", 1000**3),
    ("T", 1000**4),
    ("P", 1000**5),
    ("E", 1000**6),
)


def parse_memory_bytes(value: str) -> float:
    """Parse a Kubernetes memory quantity to bytes.

    A bare number is bytes, per the Kubernetes quantity grammar -- this is the
    form the apiserver itself returns for ``node.status.allocatable.memory`` on
    some distributions, so getting it wrong is not a corner case.

    Args:
        value: Memory string like '256Mi', '1Gi', '1024Ki', '8G', '134217728'.

    Returns:
        Memory in bytes.

    Raises:
        ValueError: The value is not a recognizable Kubernetes quantity.
    """
    value = value.strip()
    if not value or value == "0":
        return 0.0
    for suffix, multiplier in _MEMORY_SUFFIXES:
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier
    return float(value)


def parse_memory_mib(value: str) -> int:
    """Parse Kubernetes memory string to MiB (int).

    Shares :func:`parse_memory_bytes` with :func:`parse_memory_gib`. The two
    used to disagree on a bare number -- this one read it as MiB while the
    other read it as bytes -- so AIPERF_K8S_WORKER_POD_MEMORY given in bytes
    passed preflight and then requested a petabyte-scale limit per container,
    leaving every worker pod unschedulable. This one also had no decimal
    (G/M/T) branch at all and raised ValueError on '8G'.

    Args:
        value: Memory string like '256Mi', '1Gi', '1024Ki', '8G'.

    Returns:
        Memory value in MiB, rounded down but never to zero for a nonzero
        input (a sub-MiB request still needs a page).
    """
    total_bytes = parse_memory_bytes(value)
    if total_bytes <= 0:
        return 0
    return max(1, int(total_bytes / 1024**2))


def parse_memory_gib(value: str) -> float:
    """Parse Kubernetes memory string to GiB (float).

    Args:
        value: Memory string like '1Gi', '512Mi', '1024M'.

    Returns:
        Memory in GiB.
    """
    return parse_memory_bytes(value) / 1024**3


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

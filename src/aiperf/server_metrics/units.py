# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect units from prometheus metric descriptions and names"""

import contextlib
import fnmatch
import re
from functools import lru_cache

from aiperf.common.enums import (
    BaseMetricUnit,
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricOverTimeUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    PowerMetricUnit,
    TemperatureMetricUnit,
)

# Mapping of Prometheus metric name suffixes to unit strings
# Note: Suffixes are sorted by length (longest first) for matching,
# so compound suffixes like _tokens_total will match before _total
_METRIC_SUFFIX_TO_UNIT: dict[str, BaseMetricUnit] = {
    # Time units
    "_seconds": MetricTimeUnit.SECONDS,
    "_seconds_total": MetricTimeUnit.SECONDS,
    "_milliseconds": MetricTimeUnit.MILLISECONDS,
    "_ms": MetricTimeUnit.MILLISECONDS,
    "_ms_total": MetricTimeUnit.MILLISECONDS,
    "_nanoseconds": MetricTimeUnit.NANOSECONDS,
    "_ns": MetricTimeUnit.NANOSECONDS,
    "_ns_total": MetricTimeUnit.NANOSECONDS,
    # Size/data units
    "_bytes": MetricSizeUnit.BYTES,
    "_kilobytes": MetricSizeUnit.KILOBYTES,
    "_megabytes": MetricSizeUnit.MEGABYTES,
    "_gigabytes": MetricSizeUnit.GIGABYTES,
    "_bytes_total": MetricSizeUnit.BYTES,
    # Count/quantity units
    "_total": GenericMetricUnit.COUNT,
    "_count": GenericMetricUnit.COUNT,
    "_tokens": GenericMetricUnit.TOKENS,
    "_tokens_total": GenericMetricUnit.TOKENS,
    "_requests": GenericMetricUnit.REQUESTS,
    "_requests_total": GenericMetricUnit.REQUESTS,
    "request_success": GenericMetricUnit.REQUESTS,
    "_errors": GenericMetricUnit.ERRORS,
    "_errors_total": GenericMetricUnit.ERRORS,
    "_error_count": GenericMetricUnit.ERRORS,
    "_error_count_total": GenericMetricUnit.ERRORS,
    "_reqs": GenericMetricUnit.REQUESTS,
    "_blocks": GenericMetricUnit.BLOCKS,
    "_blocks_total": GenericMetricUnit.BLOCKS,
    "_block_count": GenericMetricUnit.BLOCKS,
    # Rate units (unambiguous suffixes only)
    "_gb_s": MetricOverTimeUnit.GB_PER_SECOND,
    # Ratio/percentage units
    "_ratio": GenericMetricUnit.RATIO,
    "_percent": GenericMetricUnit.PERCENT,
    "_perc": GenericMetricUnit.PERCENT,
    # Physical units
    "_celsius": TemperatureMetricUnit.CELSIUS,
    "_joules": EnergyMetricUnit.JOULE,
    "_watts": PowerMetricUnit.WATT,
}  # fmt: skip

# Wildcard patterns for metric name matching (supports * and ? globs)
# Sorted by specificity (longest pattern first) for matching priority
_METRIC_WILDCARD_TO_UNIT: dict[str, BaseMetricUnit] = {
    "*num_requests_*": GenericMetricUnit.REQUESTS,
}  # fmt: skip

# Pre-compute sorted suffixes (longest first) for efficient matching
_SORTED_SUFFIXES = sorted(_METRIC_SUFFIX_TO_UNIT.keys(), key=len, reverse=True)
_SORTED_WILDCARDS = sorted(_METRIC_WILDCARD_TO_UNIT.keys(), key=len, reverse=True)


# =============================================================================
# Public API
# =============================================================================


@lru_cache(maxsize=2048)
def infer_unit(
    metric_name: str,
    description: str | None = None,
) -> BaseMetricUnit | None:
    """Infer the unit for a metric using multiple sources with priority ordering.

    Uses a multi-source inference strategy to determine metric units from both
    metric names and descriptions. This is crucial for proper display formatting
    and unit conversion in exports.

    Priority order:
    1. Scale from description (ratio vs percent range indicators)
       - Looks for explicit ranges like "(0-1)" or "(0-100)"
       - Most authoritative source when present
    2. Unit from description (explicit "in seconds", etc.)
       - Parses natural language unit specifications
       - Handles both long form ("in seconds") and abbreviated ("(ms)")
    3. Unit from metric name suffix
       - Standard Prometheus naming conventions
       - Common suffixes like "_seconds", "_bytes", "_total"

    The scale detection (step 1) can override suffix-based inference when
    a metric has a suffix like "_percent" but the description indicates
    a 0-1 range (which should be "ratio", not "percent").

    Cached with LRU cache (2048 entries) for performance since metrics are
    processed repeatedly during collection.

    Args:
        metric_name: Full metric name (e.g., 'request_duration_seconds', 'cache_hit_rate')
        description: Optional description text from metric HELP text

    Returns:
        Inferred BaseMetricUnit enum value, or None if no unit can be determined.
        Common returns: MetricTimeUnit.SECONDS, MetricSizeUnit.BYTES,
        GenericMetricUnit.PERCENT, etc.

    Examples:
        >>> # Suffix-based inference
        >>> infer_unit("request_duration_seconds")
        MetricTimeUnit.SECONDS

        >>> # Description-based inference
        >>> infer_unit("latency", "Request latency in milliseconds")
        MetricTimeUnit.MILLISECONDS

        >>> # Scale override from description
        >>> infer_unit("cache_hit_percent", "Hit rate (0.0-1.0)")
        GenericMetricUnit.RATIO  # Description overrides "_percent" suffix

        >>> # Complex description parsing
        >>> infer_unit("gpu_power", "Power consumption (in W)")
        PowerMetricUnit.WATT

        >>> # No unit determinable
        >>> infer_unit("unknown_metric")
        None
    """
    # Check for explicit scale indicators in description first
    # This takes priority because it's the most authoritative source
    if scale := _parse_scale_from_description(description):
        return scale

    # Check for explicit unit in description
    if desc_unit := _parse_unit_from_description(description):
        return desc_unit

    # Fall back to suffix-based inference
    return _parse_unit_from_metric_name(metric_name)


# =============================================================================
# Private API
# =============================================================================


def _parse_unit_from_metric_name(metric_name: str) -> BaseMetricUnit | None:
    """Infer unit from Prometheus metric name suffix or wildcard pattern.

    Prometheus naming conventions use suffixes like _seconds, _bytes, _total
    to indicate the unit of a metric. Additionally supports glob-style wildcard
    patterns (*, ?) for matching metric name prefixes or complex patterns.

    Args:
        metric_name: Full metric name (e.g., 'request_duration_seconds')

    Returns:
        BaseMetricUnit if a recognized suffix or pattern is found, None otherwise
    """
    name_lower = metric_name.lower()

    # Check longer suffixes first to avoid partial matches
    # e.g., "_milliseconds" should match before "_seconds"
    with contextlib.suppress(StopIteration):
        return next(
            _METRIC_SUFFIX_TO_UNIT[suffix]
            for suffix in _SORTED_SUFFIXES
            if name_lower.endswith(suffix)
        )

    # Fast path: check simple containment patterns before expensive fnmatch
    # Pattern "*num_requests_*" is equivalent to "num_requests_" in name
    if "num_requests_" in name_lower:
        return GenericMetricUnit.REQUESTS

    # Check remaining wildcard patterns (fnmatch supports * and ? globs)
    with contextlib.suppress(StopIteration):
        return next(
            _METRIC_WILDCARD_TO_UNIT[pattern]
            for pattern in _SORTED_WILDCARDS
            if fnmatch.fnmatch(name_lower, pattern)
        )

    return None


# Description-based unit patterns
# Maps regex patterns to BaseMetricUnit. Patterns are checked in order.
# All patterns are case-insensitive.
# Optimizations: Consolidated "in X"+"(X)" (21→11), then abbrev+full (11→9)
_DESCRIPTION_UNIT_PATTERNS: list[tuple[re.Pattern[str], BaseMetricUnit]] = [
    # Time units: "in seconds", "(seconds)", "in ms/milliseconds", "in ns/nanoseconds"
    (re.compile(r"(?:\bin\s+|\()seconds?(?:\b|\))", re.IGNORECASE), MetricTimeUnit.SECONDS),
    (re.compile(r"(?:\bin\s+|\()(?:milliseconds?|ms)(?:\b|\))", re.IGNORECASE), MetricTimeUnit.MILLISECONDS),
    (re.compile(r"(?:\bin\s+|\()(?:nanoseconds?|ns)(?:\b|\))", re.IGNORECASE), MetricTimeUnit.NANOSECONDS),
    # Size units: "in bytes", "(bytes)"
    (re.compile(r"(?:\bin\s+|\()bytes?(?:\b|\))", re.IGNORECASE), MetricSizeUnit.BYTES),
    # Rate units: "in GB/s", "(GB/s)", "in tokens/s", "(tokens/s)", etc.
    (re.compile(r"(?:\bin\s+|\()GB/s(?:\b|\))", re.IGNORECASE), MetricOverTimeUnit.GB_PER_SECOND),
    (re.compile(r"(?:\bin\s+|\()MB/s(?:\b|\))", re.IGNORECASE), MetricOverTimeUnit.MB_PER_SECOND),
    (re.compile(r"(?:\bin\s+|\()tokens?/s(?:ec(?:ond)?)?(?:\b|\))", re.IGNORECASE), MetricOverTimeUnit.TOKENS_PER_SECOND),
    (re.compile(r"(?:\bin\s+|\()requests?/s(?:ec(?:ond)?)?(?:\b|\))", re.IGNORECASE), MetricOverTimeUnit.REQUESTS_PER_SECOND),
    # Generic units: "in tokens" (no parenthetical form needed)
    (re.compile(r"\bin\s+tokens?\b", re.IGNORECASE), GenericMetricUnit.TOKENS),
]  # fmt: skip

# Generic mapping from unit tags/abbreviations to BaseMetricUnit.
# Used by _parse_parenthetical_unit to handle "(in <unit>)" patterns generically.
# Keys are case-sensitive where needed (e.g., "mJ" vs "MJ").
_UNIT_TAG_TO_UNIT: dict[str, BaseMetricUnit] = {
    # Size units (binary)
    "MiB": MetricSizeUnit.MEGABYTES,  # Mebibytes (1024^2), mapped to MEGABYTES which uses same value
    "GiB": MetricSizeUnit.GIGABYTES,
    "KiB": MetricSizeUnit.KILOBYTES,
    "B": MetricSizeUnit.BYTES,
    # Size units (decimal)
    "MB": MetricSizeUnit.MEGABYTES,
    "GB": MetricSizeUnit.GIGABYTES,
    "KB": MetricSizeUnit.KILOBYTES,
    "TB": MetricSizeUnit.TERABYTES,
    # Temperature units
    "C": TemperatureMetricUnit.CELSIUS,
    "°C": TemperatureMetricUnit.CELSIUS,
    "F": TemperatureMetricUnit.FAHRENHEIT,
    "°F": TemperatureMetricUnit.FAHRENHEIT,
    "K": TemperatureMetricUnit.KELVIN,
    # Frequency units
    "Hz": FrequencyMetricUnit.HERTZ,
    "MHz": FrequencyMetricUnit.MEGAHERTZ,
    "GHz": FrequencyMetricUnit.GIGAHERTZ,
    # Power units
    "W": PowerMetricUnit.WATT,
    "mW": PowerMetricUnit.MILLIWATT,
    # Energy units
    "J": EnergyMetricUnit.JOULE,
    "mJ": EnergyMetricUnit.MILLIJOULE,
    "MJ": EnergyMetricUnit.MEGAJOULE,
    # Time units
    "s": MetricTimeUnit.SECONDS,
    "sec": MetricTimeUnit.SECONDS,
    "ms": MetricTimeUnit.MILLISECONDS,
    "ns": MetricTimeUnit.NANOSECONDS,
    "us": MetricTimeUnit.MICROSECONDS,
    "µs": MetricTimeUnit.MICROSECONDS,
    # Rate units
    "GB/s": MetricOverTimeUnit.GB_PER_SECOND,
    "MB/s": MetricOverTimeUnit.MB_PER_SECOND,
    # Percentage/ratio units
    "%": GenericMetricUnit.PERCENT,
    "percent": GenericMetricUnit.PERCENT,
}  # fmt: skip

# Regex to match "(in <unit>)" pattern and capture the unit.
# Examples: "(in MiB)", "(in W)", "(in mJ)", "(in C)"
# Use [^\s)]+ instead of [^)]+ to avoid ReDoS from overlapping quantifiers
_PARENTHETICAL_IN_UNIT_PATTERN = re.compile(r"\(in\s+([^\s)]+)\)")


def _parse_parenthetical_unit(description: str | None) -> BaseMetricUnit | None:
    """Parse unit from "(in <unit>)" pattern in description.

    This is a generic parser for DCGM-style metric descriptions that use
    patterns like "(in MiB)", "(in W)", "(in mJ)", "(in C)".

    Only exact case-sensitive matches are supported to avoid ambiguity
    with units that differ only in case (e.g., mJ vs MJ).

    Args:
        description: Metric description text, or None

    Returns:
        BaseMetricUnit if a recognized unit is found, None otherwise
    """
    if not description:
        return None

    if match := _PARENTHETICAL_IN_UNIT_PATTERN.search(description):
        unit_str = match.group(1).strip()
        return _UNIT_TAG_TO_UNIT.get(unit_str)

    return None


# Patterns for detecting ratio (0-1 scale) vs percent (0-100 scale)
# These override suffix-based inference when present in descriptions.
# Order matters: check ratio patterns first since they're more specific.
_RATIO_RANGE_PATTERN = re.compile(
    r"\(0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\)"  # (0-1), (0.0-1.0), (0 to 1)
    r"|\b0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\b"  # 0-1, 0.0-1.0 without parens
    r"|\brange\s+0(?:\.0)?\s*(?:[-–—]+|to)\s*1(?:\.0)?\b"  # range 0-1
    r"|(?:\b|\()1(?:\.0)?\s*(?:means|is|equals?|==?)\s*100\s*(?:%|percent)",  # 1=100%
    re.IGNORECASE,
)

_PERCENT_RANGE_PATTERN = re.compile(
    r"\(0(?:\.0)?\s*(?:[-–—]+|to)\s*100(?:\.0)?\)"  # (0-100), (0.0-100.0), (0 to 100)
    r"|\brange\s+0\s*(?:[-–—]+|to)\s*100\b"  # range 0-100
    r"|\b0\s*(?:[-–—]+|to)\s*100\s*%",  # 0-100%, 0 to 100%
    re.IGNORECASE,
)


def _parse_scale_from_description(description: str | None) -> BaseMetricUnit | None:
    """Detect ratio vs percent scale from description range indicators.

    This function looks for explicit range indicators in descriptions to
    distinguish between:
    - ratio: Values in [0.0, 1.0] range (statistical convention)
    - percent: Values in [0, 100] range

    Args:
        description: Metric description text, or None

    Returns:
        GenericMetricUnit.RATIO if 0-1 range detected, GenericMetricUnit.PERCENT if 0-100 range detected,
        None if no range indicator found
    """
    if not description:
        return None

    # Check ratio pattern first (more specific)
    if _RATIO_RANGE_PATTERN.search(description):
        return GenericMetricUnit.RATIO

    # Check percent pattern
    if _PERCENT_RANGE_PATTERN.search(description):
        return GenericMetricUnit.PERCENT

    return None


def _parse_unit_from_description(description: str | None) -> BaseMetricUnit | None:
    """Extract unit from metric description text.

    Looks for explicit unit mentions like "in seconds", "(tokens/s)", etc.
    This is more authoritative than suffix-based inference since descriptions
    often contain the exact unit specification.

    Priority order:
    1. Generic "(in <unit>)" patterns (DCGM-style)
    2. Specific regex patterns for common phrases

    Args:
        description: Metric description text, or None

    Returns:
        BaseMetricUnit if a recognized pattern is found, None otherwise
    """
    if not description:
        return None

    # Try generic "(in <unit>)" pattern first (handles DCGM metrics)
    if unit := _parse_parenthetical_unit(description):
        return unit

    # Fall back to specific regex patterns
    with contextlib.suppress(StopIteration):
        return next(
            unit
            for pattern, unit in _DESCRIPTION_UNIT_PATTERNS
            if pattern.search(description)
        )

    return None

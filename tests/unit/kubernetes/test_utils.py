# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.utils module."""

import pytest
from pytest import param

from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
    parse_memory_mib,
)


class TestParseCpu:
    """Tests for parse_cpu function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            param("100m", 0.1, id="100-millicores"),
            param("500m", 0.5, id="500-millicores"),
            param("1000m", 1.0, id="1000-millicores"),
            param("1500m", 1.5, id="1500-millicores"),
            param("10m", 0.01, id="10-millicores"),
            param("0m", 0.0, id="zero-millicores"),
            param("1", 1.0, id="one-core"),
            param("2.5", 2.5, id="fractional-cores"),
            param("4", 4.0, id="four-cores"),
            param("0.25", 0.25, id="quarter-core"),
            param("0", 0.0, id="zero-string"),
            param("", 0.0, id="empty-string"),
        ],
    )  # fmt: skip
    def test_parse_cpu(self, value: str, expected: float) -> None:
        """Test CPU value parsing from Kubernetes format."""
        assert parse_cpu(value) == pytest.approx(expected)


class TestParseMemoryMib:
    """Tests for parse_memory_mib function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            param("256Mi", 256, id="256-mib"),
            param("512Mi", 512, id="512-mib"),
            param("1Gi", 1024, id="1-gib"),
            param("2Gi", 2048, id="2-gib"),
            param("0.5Gi", 512, id="half-gib"),
            param("1.5Gi", 1536, id="1.5-gib"),
            param("1024Ki", 1, id="kibibytes"),
            param("2048Ki", 2, id="2048-kib"),
            param("0Mi", 0, id="zero-mib"),
            param("100", 100, id="plain-number"),
            param("0", 0, id="zero-string"),
            param("", 0, id="empty-string"),
        ],
    )  # fmt: skip
    def test_parse_memory_mib(self, value: str, expected: int) -> None:
        """Test memory value parsing to MiB."""
        assert parse_memory_mib(value) == expected


class TestParseMemoryGib:
    """Tests for parse_memory_gib function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            param("1Gi", 1.0, id="1-gib"),
            param("2Gi", 2.0, id="2-gib"),
            param("0.5Gi", 0.5, id="half-gib"),
            param("512Mi", 0.5, id="512-mib"),
            param("256Mi", 0.25, id="256-mib"),
            param("1024Mi", 1.0, id="1024-mib"),
            param("1G", 1000 / 1024, id="1-gb-decimal"),
            param("1M", 1 / 1024, id="1-mb-decimal"),
            param("1024Ki", 1 / 1024, id="1024-kib"),
            param("0", 0.0, id="zero-string"),
            param("", 0.0, id="empty-string"),
        ],
    )  # fmt: skip
    def test_parse_memory_gib(self, value: str, expected: float) -> None:
        """Test memory value parsing to GiB."""
        assert parse_memory_gib(value) == pytest.approx(expected)


class TestFormatCpu:
    """Tests for format_cpu function."""

    @pytest.mark.parametrize(
        "cores,expected",
        [
            param(0.1, "100m", id="100-millicores"),
            param(0.5, "500m", id="500-millicores"),
            param(0.25, "250m", id="250-millicores"),
            param(1.0, "1.0", id="one-core"),
            param(2.5, "2.5", id="2.5-cores"),
            param(4.0, "4.0", id="four-cores"),
        ],
    )  # fmt: skip
    def test_format_cpu(self, cores: float, expected: str) -> None:
        """Test CPU formatting for display."""
        assert format_cpu(cores) == expected


class TestFormatMemory:
    """Tests for format_memory function."""

    @pytest.mark.parametrize(
        "gib,expected",
        [
            param(0.5, "512Mi", id="half-gib"),
            param(0.25, "256Mi", id="quarter-gib"),
            param(1.0, "1.0Gi", id="one-gib"),
            param(2.0, "2.0Gi", id="two-gib"),
            param(1.5, "1.5Gi", id="1.5-gib"),
        ],
    )  # fmt: skip
    def test_format_memory(self, gib: float, expected: str) -> None:
        """Test memory formatting for display."""
        assert format_memory(gib) == expected


class TestRoundTrip:
    """Tests verifying parse/format round-trip consistency."""

    @pytest.mark.parametrize(
        "cores",
        [0.1, 0.5, 1.0, 2.0, 4.0],
    )
    def test_cpu_round_trip_format_then_parse(self, cores: float) -> None:
        """Test format_cpu -> parse_cpu preserves value."""
        formatted = format_cpu(cores)
        assert parse_cpu(formatted) == pytest.approx(cores)

    @pytest.mark.parametrize(
        "gib",
        [0.25, 0.5, 1.0, 2.0],
    )
    def test_memory_round_trip_format_then_parse(self, gib: float) -> None:
        """Test format_memory -> parse_memory_gib preserves value."""
        formatted = format_memory(gib)
        assert parse_memory_gib(formatted) == pytest.approx(gib)

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
            # A bare number is BYTES per the Kubernetes quantity grammar --
            # this used to be read as MiB here while parse_memory_gib read the
            # same string as bytes, so the two disagreed by 2**20.
            param("104857600", 100, id="plain-number-is-bytes"),
            param("100", 1, id="sub-mib-never-rounds-to-zero"),
            param("8G", 7629, id="decimal-gigabytes"),
            param("1Ti", 1024 * 1024, id="tebibytes"),
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
            # Decimal suffixes are powers of 1000 BYTES: 1G is 1e9 bytes,
            # which is 0.9313 GiB. The old expectation (1000/1024) mixed the
            # two bases and overstated every decimal quantity by ~7%.
            param("1G", 1e9 / 1024**3, id="1-gb-decimal"),
            param("1M", 1e6 / 1024**3, id="1-mb-decimal"),
            param("104857600", 0.09765625, id="plain-number-is-bytes"),
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


class TestMemoryParsersAgree:
    """The MiB and GiB parsers must never disagree about the same string.

    They were independent implementations: one read a bare number as MiB, the
    other as bytes, and only the GiB one knew the decimal (G/M) suffixes. An
    AIPERF_K8S_WORKER_POD_MEMORY expressed in bytes therefore passed preflight
    (correct parser) and then produced a petabyte-scale container request
    (wrong parser), leaving every worker pod unschedulable.
    """

    @pytest.mark.parametrize(
        "value",
        [
            param("256Mi", id="mebibytes"),
            param("1Gi", id="gibibytes"),
            param("1024Ki", id="kibibytes"),
            param("8589934592", id="bare-bytes"),
            param("8G", id="decimal-gigabytes"),
            param("512M", id="decimal-megabytes"),
            param("1Ti", id="tebibytes"),
            param("0", id="zero"),
        ],
    )  # fmt: skip
    def test_mib_and_gib_describe_the_same_quantity(self, value: str) -> None:
        from aiperf.kubernetes.utils import parse_memory_bytes

        expected_mib = parse_memory_bytes(value) / 1024**2
        assert parse_memory_gib(value) == pytest.approx(expected_mib / 1024)
        if expected_mib >= 1:
            # parse_memory_mib floors to an int, so allow the truncated MiB.
            assert 0 <= expected_mib - parse_memory_mib(value) < 1

    def test_decimal_suffix_no_longer_raises(self) -> None:
        """parse_memory_mib had no G/M/T branch and died on '8G'."""
        assert parse_memory_mib("8G") > 0


class TestRequestsProgressPercent:
    """`aiperf kube list` must report the phase the job is actually in.

    status.phases is a CRD object map and the apiserver alphabetizes those
    keys on storage, so "last in iteration order" resolved to `warmup`, not to
    the newest phase: a job 20% into profiling printed 100%.
    """

    def _phases(self) -> dict:
        # Alphabetized exactly as the apiserver returns them.
        return {
            "profiling": {"requestsProgressPercent": 20.0},
            "warmup": {"requestsProgressPercent": 100.0},
        }

    def test_uses_current_phase(self) -> None:
        from aiperf.kubernetes.models import _requests_progress_percent

        assert _requests_progress_percent(self._phases(), "profiling") == 20.0

    def test_falls_back_when_current_phase_missing(self) -> None:
        from aiperf.kubernetes.models import _requests_progress_percent

        assert _requests_progress_percent(self._phases(), None) == 100.0

    def test_falls_back_when_current_phase_has_no_percent(self) -> None:
        from aiperf.kubernetes.models import _requests_progress_percent

        phases = {"profiling": {}, "warmup": {"requestsProgressPercent": 100.0}}
        assert _requests_progress_percent(phases, "profiling") == 100.0

    def test_none_when_no_phase_reports_progress(self) -> None:
        from aiperf.kubernetes.models import _requests_progress_percent

        assert _requests_progress_percent({"warmup": {}}, "warmup") is None

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags


class TestMetricFlags:
    @pytest.mark.parametrize(
        "flags, flags_to_check, expected",
        [
            (MetricFlags.NONE, MetricFlags.NONE, True),
            (MetricFlags.NONE, MetricFlags.STREAMING_ONLY, False),
            (MetricFlags.NONE, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
            ###
            (MetricFlags.STREAMING_ONLY, MetricFlags.NONE, True),
            (MetricFlags.STREAMING_ONLY, MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.STREAMING_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
            (MetricFlags.STREAMING_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, False),
            ###
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, False),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, True),
            ###
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, True),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, True),
        ],
    )  # fmt: skip
    def test_has_flags(self, flags, flags_to_check, expected):
        assert flags.has_flags(flags_to_check) == expected, (
            f"Expected {flags}.has_flags({flags_to_check}) to equal {expected}"
        )

    @pytest.mark.parametrize(
        "flags, flags_to_check, expected",
        [
            (MetricFlags.NONE, MetricFlags.NONE, True),
            (MetricFlags.NONE, MetricFlags.NONE | MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.NONE, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, True),
            ###
            (MetricFlags.STREAMING_ONLY, MetricFlags.NONE, True),
            (MetricFlags.STREAMING_ONLY, MetricFlags.STREAMING_ONLY, False),
            ###
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, False),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, False),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_TOKENS_ONLY, False),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.NONE, True),
            ###
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.NONE, True),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
        ],
    )  # fmt: skip
    def test_missing_flags(self, flags, flags_to_check, expected):
        assert flags.missing_flags(flags_to_check) == expected, (
            f"Expected {flags}.missing_flags({flags_to_check}) to equal {expected}"
        )

    @pytest.mark.parametrize(
        "flags, flags_to_check, expected",
        [
            (MetricFlags.NONE, MetricFlags.NONE, False),
            (MetricFlags.NONE, MetricFlags.STREAMING_ONLY, False),
            (MetricFlags.NONE, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, False),
            ###
            (MetricFlags.STREAMING_ONLY, MetricFlags.NONE, False),
            (MetricFlags.STREAMING_ONLY, MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.STREAMING_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, True),
            (MetricFlags.STREAMING_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, False),
            ###
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, False),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, True),
            (MetricFlags.PRODUCES_TOKENS_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, True),
            ###
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY, True),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.STREAMING_ONLY | MetricFlags.PRODUCES_TOKENS_ONLY, True),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.PRODUCES_TOKENS_ONLY, True),
            (MetricFlags.STREAMING_TOKENS_ONLY, MetricFlags.NONE, False),
            ###
            (MetricFlags.NO_CONSOLE, MetricFlags.NO_CONSOLE, True),
            (MetricFlags.NO_CONSOLE, MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL, True),
            (MetricFlags.NO_CONSOLE, MetricFlags.INTERNAL, False),
            (MetricFlags.NO_CONSOLE, MetricFlags.EXPERIMENTAL, False),
            ###
            (MetricFlags.INTERNAL, MetricFlags.INTERNAL, True),
            (MetricFlags.INTERNAL, MetricFlags.NO_CONSOLE, False),
            (MetricFlags.INTERNAL, MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL, True),
            ###
            (MetricFlags.EXPERIMENTAL, MetricFlags.EXPERIMENTAL, True),
            (MetricFlags.EXPERIMENTAL, MetricFlags.NO_CONSOLE, False),
            (MetricFlags.EXPERIMENTAL, MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL, True),
        ],
    )  # fmt: skip
    def test_has_any_flags(self, flags, flags_to_check, expected):
        assert flags.has_any_flags(flags_to_check) == expected, (
            f"Expected {flags}.has_any_flags({flags_to_check}) to equal {expected}"
        )

    def test_internal_does_not_inherit_no_console(self):
        """Test that INTERNAL flag no longer inherits NO_CONSOLE"""
        assert MetricFlags.INTERNAL.missing_flags(MetricFlags.NO_CONSOLE)

    def test_experimental_does_not_inherit_no_console(self):
        """Test that EXPERIMENTAL flag no longer inherits NO_CONSOLE"""
        assert MetricFlags.EXPERIMENTAL.missing_flags(MetricFlags.NO_CONSOLE)

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

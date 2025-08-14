# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.ui.utils import format_bytes, format_elapsed_time, format_eta

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize(
    "bytes, expected",
    [
        (None, "--"),
        (0, "0 B"),
        (1, "1 B"),
        (999, "999 B"),
        (1000, "1.0 KB"),  # 0.976 rounded to 1.0
        (1023, "1.0 KB"),
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (1024**5, "1.0 PB"),
        (1024**6, "1.0 EB"),
        (1024**7, "1.0 ZB"),
        (1024**8, "1.0 YB"),
    ],
)
def test_format_bytes(bytes, expected):
    assert format_bytes(bytes) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "--"),
        (0, "0.0s"),
        (1, "1.0s"),
        (0.5, "0.5s"),
        (0.9, "0.9s"),
        (0.999, "1.0s"),
        (1.001, "1.0s"),
        (1.5, "1.5s"),
        (1.999, "2.0s"),
        (2.001, "2.0s"),
        (60, "1m"),
        (60 * 60 - 1, "59m 59s"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h"),
        (60 * 60 + 69, "1h 1m"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 * 365, "365d"),
    ],
)
def test_format_eta(seconds, expected) -> None:
    assert format_eta(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "--"),
        (0, "0.0s"),
        (1, "1.0s"),
        (0.5, "0.5s"),
        (0.9, "0.9s"),
        (0.999, "1.0s"),
        (1.001, "1.0s"),
        (1.5, "1.5s"),
        (1.999, "2.0s"),
        (2.001, "2.0s"),
        (60, "1m"),
        (60 * 60 - 1, "59m 59s"),
        (60 * 60, "1h"),
        (60 * 60 + 1, "1h 1s"),
        (60 * 60 + 69, "1h 1m 9s"),
        (60 * 60 * 24, "1d"),
        (60 * 60 * 24 * 365, "365d"),
    ],
)
def test_format_elapsed_time(seconds, expected) -> None:
    assert format_elapsed_time(seconds) == expected

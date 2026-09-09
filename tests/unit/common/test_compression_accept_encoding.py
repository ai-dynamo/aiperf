# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pytest import param

from aiperf.common.compression import parse_accept_encoding


@pytest.mark.parametrize(
    "header,expected",
    [
        param("gzip", {"gzip": 1.0}, id="single"),
        param("gzip;q=0.5, br", {"gzip": 0.5, "br": 1.0}, id="qvalues"),
        param("", {}, id="empty"),
    ],
)  # fmt: skip
def test_parse_accept_encoding(header, expected):
    assert parse_accept_encoding(header) == expected

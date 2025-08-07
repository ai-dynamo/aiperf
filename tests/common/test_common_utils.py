# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.utils import close_enough


class TestCommonUtils:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (1, 1, True),
            (1, 1.0, True),
            (1, 1.0000000001, True),
            (1, 1.000000002, False),
            (1, 2, False),
            (1, 2.0, False),
            (1, 2.0000000002, False),
            (1, 2.0000000003, False),
        ],
    )
    def test_close_enough(self, a, b, expected):
        assert close_enough(a, b) == expected, (
            f"close_enough({a}, {b}) should be {expected}, but was {close_enough(a, b)}"
        )

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1, 2, 3], [1, 2, 3], True),
            ([1, 2, 3], [1, 2, 3.0], True),
            ([1, 2, 3], [1, 2, 3.0000000001], True),
            ([1, 2, 3], [1, 2, 3.000000002], False),
            ([1, 2, 3], [1, 2, 4], False),
            ([1, 2, 3], [1, 2, 4.0], False),
            ([1, 2, 3], [1, 2, 4.0000000002], False),
            ([1, 2, 3], [1, 2, 4.0000000003], False),
            ([4, 3, 2, 1], [1, 2, 3, 4], False),
        ],
    )
    def test_close_enough_list(self, a, b, expected):
        assert close_enough(a, b) == expected, (
            f"close_enough({a}, {b}) should be {expected}, but was {close_enough(a, b)}"
        )

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (1, [1, 2, 3], False),
            ([1, 2, 3], 1, False),
            (1, [1, 1, 1.0, 1.0000000001], True),
            ([25, 25, 25.0, 25.0000000001], 25, True),
        ],
    )
    def test_close_enough_single_vs_list(self, a, b, expected):
        assert close_enough(a, b) == expected, (
            f"close_enough({a}, {b}) should be {expected}, but was {close_enough(a, b)}"
        )

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1, 2, 3], [1, 2, 3, 4]),
            ([1, 2, 3], [1, 2, 3, 4.0]),
        ],
    )
    def test_close_enough_list_different_length(self, a, b):
        with pytest.raises(ValueError):
            close_enough(a, b)

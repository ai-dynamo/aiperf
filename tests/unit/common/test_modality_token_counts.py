# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models.modality_token_counts import ModalityTokenCounts


class TestModalityTokenCounts:
    def test_total_sums_non_none_fields(self):
        counts = ModalityTokenCounts(text=100, image=200)
        assert counts.total == 300

    def test_total_with_all_fields(self):
        counts = ModalityTokenCounts(text=10, image=20, video=30, audio=40)
        assert counts.total == 100

    def test_total_all_none_returns_zero(self):
        counts = ModalityTokenCounts()
        assert counts.total == 0

    def test_total_single_field(self):
        counts = ModalityTokenCounts(image=500)
        assert counts.total == 500

    def test_scale_to_proportional(self):
        counts = ModalityTokenCounts(text=120, image=480)
        scaled = counts.scale_to(650)
        assert scaled.text + scaled.image == 650
        assert scaled.text == 130  # round(650 * 120 / 600)
        assert scaled.image == 520  # remainder

    def test_scale_to_same_total(self):
        counts = ModalityTokenCounts(text=100, image=200)
        scaled = counts.scale_to(300)
        assert scaled.text == 100
        assert scaled.image == 200

    def test_scale_to_zero_total_returns_empty(self):
        counts = ModalityTokenCounts()
        scaled = counts.scale_to(100)
        assert scaled.total == 0

    def test_scale_to_single_modality(self):
        counts = ModalityTokenCounts(text=100)
        scaled = counts.scale_to(200)
        assert scaled.text == 200

    def test_scale_to_preserves_none_fields(self):
        counts = ModalityTokenCounts(text=100, image=200)
        scaled = counts.scale_to(600)
        assert scaled.video is None
        assert scaled.audio is None

    def test_scale_to_remainder_absorbed_by_largest(self):
        counts = ModalityTokenCounts(text=1, image=1, video=1)
        scaled = counts.scale_to(10)
        assert scaled.total == 10

    def test_equality(self):
        a = ModalityTokenCounts(text=100, image=200)
        b = ModalityTokenCounts(text=100, image=200)
        assert a == b

    def test_inequality(self):
        a = ModalityTokenCounts(text=100, image=200)
        b = ModalityTokenCounts(text=100, image=300)
        assert a != b

    @pytest.mark.parametrize(
        "local_total,target",
        [(600, 650), (100, 1000), (1000, 100), (1, 1)],
    )
    def test_scale_to_always_sums_to_target(self, local_total, target):
        counts = ModalityTokenCounts(
            text=local_total // 3, image=local_total - local_total // 3
        )
        scaled = counts.scale_to(target)
        assert scaled.total == target

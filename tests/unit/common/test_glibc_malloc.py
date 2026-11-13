# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pytest import param

from aiperf.common import glibc_malloc
from aiperf.common.constants import IS_LINUX


@pytest.mark.parametrize(
    "call",
    [
        param(glibc_malloc.pin_malloc_thresholds, id="pin_thresholds"),
        param(glibc_malloc.trim_heap, id="trim_heap"),
    ],
)  # fmt: skip
def test_glibc_malloc_call_succeeds_on_linux_and_noops_elsewhere(call) -> None:
    assert call() is IS_LINUX


def test_glibc_malloc_calls_noop_when_not_linux(monkeypatch) -> None:
    monkeypatch.setattr(glibc_malloc, "IS_LINUX", False)
    assert glibc_malloc.pin_malloc_thresholds() is False
    assert glibc_malloc.trim_heap() is False

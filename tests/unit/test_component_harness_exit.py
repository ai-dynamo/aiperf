# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest

from tests.component_integration.conftest import _component_test_os_exit


def _child_pid() -> int:
    return os.getpid()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
def test_component_exit_patch_does_not_break_forked_process_pool() -> None:
    ctx = mp.get_context("fork")

    with (
        patch("os._exit", side_effect=_component_test_os_exit),
        ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool,
    ):
        assert pool.submit(_child_pid).result(timeout=5) > 0

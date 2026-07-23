# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cleanup robustness for the parallel weka reconstruction driver's shm unlink guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aiperf.dataset.loader import weka_parallel_convert as wpc


def _run(drive_side_effect):
    """Invoke the driver with pool internals stubbed and the shm pre-unlinked so the finally block's unlink hits a missing segment."""
    captured: dict[str, object] = {}

    real_shm_cls = wpc.shared_memory.SharedMemory

    def _capturing_shm(*args, **kwargs):
        shm = real_shm_cls(*args, **kwargs)
        captured["shm"] = shm
        return shm

    def _drive(pool, tasks):
        # Reclaim the segment before the orchestrator's finally runs, so the
        # subsequent shm.unlink() in the driver sees a missing segment.
        captured["shm"].unlink()
        return drive_side_effect(pool, tasks)

    with (
        patch.object(wpc.shared_memory, "SharedMemory", side_effect=_capturing_shm),
        patch.object(wpc, "get_loader_mp_context") as mock_ctx,
        patch.object(wpc, "_drive_reconstruction_pool", side_effect=_drive),
        patch(
            "aiperf.dataset.loader.parallel_convert._shutdown_pool",
            MagicMock(),
        ),
        patch(
            "aiperf.dataset.loader.parallel_convert._ensure_valid_stdio_fds",
            MagicMock(),
        ),
        patch(
            "aiperf.dataset.loader.parallel_convert._set_daemon",
            MagicMock(),
        ),
    ):
        mock_ctx.return_value.Pool.return_value = MagicMock()
        return wpc.run_parallel_weka_reconstruction(
            tasks=[],
            tokenizer_name="test-tok",
            corpus=np.arange(8, dtype=np.int32),
            base_seed=0,
            block_size=64,
            bpe_stable_terminator_tokens=[],
            num_workers=1,
        )


def test_run_parallel_weka_reconstruction_success_tolerates_unlinked_shm():
    sentinel = [{"ok": True}]
    results = _run(lambda pool, tasks: sentinel)
    assert results is sentinel


def test_run_parallel_weka_reconstruction_unlinked_shm_does_not_mask_real_error():
    boom = RuntimeError("reconstruction blew up")

    def _raise(pool, tasks):
        raise boom

    # The original reconstruction error must propagate, not be replaced by a
    # FileNotFoundError from the cleanup unlink.
    with pytest.raises(RuntimeError) as exc_info:
        _run(_raise)
    assert exc_info.value is boom

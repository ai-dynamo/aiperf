# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A partial sweep-aggregate download must be reported, not discarded.

_download_all_sweep_operator_files returned None on the first per-file
failure, throwing away a mostly-complete aggregate directory and telling the
user nothing about which file was missing. The job-level twin has been
partial-tolerant since 5a51031db5; this copy was the exact pre-fix shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.kubernetes import results_operator_sweeps as sweeps
from aiperf.kubernetes.results_operator_common import _JobDownloadOutcome


@pytest.fixture
def listing() -> list[dict]:
    return [
        {"name": "aggregate.json"},
        {"name": "children.json"},
        {"name": ".aiperf_results_ready.json"},
    ]


@pytest.mark.asyncio
async def test_partial_download_keeps_files_and_names_the_missing(
    listing: list[dict], tmp_path: Path
) -> None:
    async def _download(_session, *, file_info, **_kwargs):
        name = file_info["name"]
        return (name, 10) if name == "aggregate.json" else None

    with (
        patch.object(
            sweeps, "_list_sweep_operator_files", AsyncMock(return_value=listing)
        ),
        patch.object(
            sweeps, "_download_sweep_operator_file", AsyncMock(side_effect=_download)
        ),
    ):
        outcome = await sweeps._download_all_sweep_operator_files(
            api_base="http://x",
            namespace="ns",
            sweep_name="s",
            output_dir=tmp_path,
            run="100",
        )

    assert isinstance(outcome, _JobDownloadOutcome)
    assert outcome.downloaded == [("aggregate.json", 10)]
    # The dot-file is refused by policy, so it is a skip, not a failure.
    assert outcome.failed == ["children.json"]
    assert outcome.complete is False


@pytest.mark.asyncio
async def test_complete_download_reports_complete(
    listing: list[dict], tmp_path: Path
) -> None:
    async def _download(_session, *, file_info, **_kwargs):
        return (file_info["name"], 10)

    with (
        patch.object(
            sweeps, "_list_sweep_operator_files", AsyncMock(return_value=listing)
        ),
        patch.object(
            sweeps, "_download_sweep_operator_file", AsyncMock(side_effect=_download)
        ),
    ):
        outcome = await sweeps._download_all_sweep_operator_files(
            api_base="http://x",
            namespace="ns",
            sweep_name="s",
            output_dir=tmp_path,
            run="100",
        )

    assert outcome.complete is True
    assert len(outcome.downloaded) == 3


@pytest.mark.asyncio
async def test_unlistable_still_returns_none(tmp_path: Path) -> None:
    """A failed listing is a hard failure -- unchanged."""
    with patch.object(
        sweeps, "_list_sweep_operator_files", AsyncMock(return_value=None)
    ):
        assert (
            await sweeps._download_all_sweep_operator_files(
                api_base="http://x",
                namespace="ns",
                sweep_name="s",
                output_dir=tmp_path,
                run="100",
            )
            is None
        )

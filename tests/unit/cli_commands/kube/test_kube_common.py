# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared `aiperf kube` helpers in ``_kube_common``.

Focused on ``resolve_child_target``, the sweep-child selector gate shared by
``kube cancel``/``logs``/``debug``/``attach`` and both ``results`` subcommands.
The selectors used to be applied only inside ``if job_id is not None``, so
``aiperf kube cancel -v 7`` silently dropped them, fell back to the last
deployed benchmark, and patched ``spec.cancel`` on the *parent* AIPerfSweep --
cancelling every variation instead of one.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.cli_commands.kube._kube_common import resolve_child_target


class TestResolveChildTargetImplicitParent:
    """A selector without an explicit job_id must hard-fail, never guess."""

    @pytest.mark.parametrize(
        "variation,trial",
        [
            param(3, None, id="variation-only"),
            param(0, None, id="variation-zero-is-still-a-selector"),
            param(3, 1, id="variation-and-trial"),
            param(None, 1, id="trial-only"),
            param(None, 0, id="trial-zero-is-still-a-selector"),
        ],
    )  # fmt: skip
    def test_resolve_child_target_selector_without_job_id_raises(
        self, variation: int | None, trial: int | None
    ) -> None:
        with pytest.raises(ValueError, match="require an explicit job_id"):
            resolve_child_target(
                None, variation=variation, trial=trial, command="kube cancel"
            )

    def test_resolve_child_target_error_names_the_invoking_command(self) -> None:
        """The message must tell the user how to re-run *this* command."""
        with pytest.raises(ValueError, match=r"aiperf kube logs <sweep-name> -v N"):
            resolve_child_target(None, variation=3, command="kube logs")

    def test_resolve_child_target_no_selector_and_no_job_id_returns_none(self) -> None:
        """The plain `aiperf kube cancel` form still falls back to the last
        deployed benchmark."""
        assert resolve_child_target(None, command="kube cancel") is None


class TestResolveChildTargetExplicitParent:
    """With an explicit parent the helper matches resolve_child_name."""

    @pytest.mark.parametrize(
        "variation,trial,expected",
        [
            param(None, None, "my-sweep", id="no-selector-passes-through"),
            param(7, None, "my-sweep-v07", id="variation"),
            param(0, None, "my-sweep-v00", id="variation-zero"),
            param(5, 0, "my-sweep-v05-t0", id="variation-and-trial-zero"),
            param(199, 9, "my-sweep-v199-t9", id="upper-bounds"),
        ],
    )  # fmt: skip
    def test_resolve_child_target_builds_the_child_name(
        self, variation: int | None, trial: int | None, expected: str
    ) -> None:
        assert (
            resolve_child_target(
                "my-sweep", variation=variation, trial=trial, command="kube cancel"
            )
            == expected
        )

    @pytest.mark.parametrize(
        "variation,trial,match",
        [
            param(None, 1, "trial requires variation", id="trial-without-variation"),
            param(200, None, "outside the supported range", id="variation-too-large"),
            param(-1, None, "outside the supported range", id="variation-negative"),
            param(1, 10, "outside the supported range", id="trial-too-large"),
        ],
    )  # fmt: skip
    def test_resolve_child_target_out_of_range_selector_raises(
        self, variation: int | None, trial: int | None, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            resolve_child_target(
                "my-sweep", variation=variation, trial=trial, command="kube cancel"
            )

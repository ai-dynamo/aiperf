# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flags only the linear replay loaders read must be refused on a graph run.

``--trace-session-sample-ratio``, ``--max-idle-gap-cap-seconds``,
``--omit-kv-hints``, ``--no-force-min-tokens`` and
``--system-idle-gap-cap-seconds`` are declared on FileDataset / the profiling
phase and consumed by ``baseten_trace`` and AGENTIC_REPLAY. Nothing under
``dataset/graph`` or ``timing`` (outside ``agentic_replay``) reads them, so on a
graph run they were accepted and silently did nothing.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import assert_ctx_knobs_supported
from aiperf.dataset.graph.adapters.dynamo.trace_reader import DynamoTraceAdapterError
from aiperf.dataset.graph.parse_context import GraphParseContext
from aiperf.dataset.graph.workload_detect import _resolve_replay_only_knobs


@pytest.mark.parametrize(
    "knobs",
    [
        param(("--omit-kv-hints",), id="one"),
        param(("--max-idle-gap-cap-seconds", "--no-force-min-tokens"), id="several"),
    ],
)  # fmt: skip
def test_replay_only_knobs_are_refused_by_name(knobs: tuple[str, ...]) -> None:
    """The refusal quotes each flag the operator typed, not a generic message."""
    with pytest.raises(DynamoTraceAdapterError) as exc:
        assert_ctx_knobs_supported(GraphParseContext(replay_only_knobs=knobs))
    for flag in knobs:
        assert flag in str(exc.value)


def test_no_knobs_set_passes() -> None:
    """A run that named none of them is untouched -- this must not fire by default."""
    assert_ctx_knobs_supported(GraphParseContext())


@pytest.mark.parametrize(
    "overrides,expected",
    [
        param({}, (), id="all-default"),
        param({"omit_kv_hints": True}, ("--omit-kv-hints",), id="bool-false-default"),
        param(
            {"force_min_tokens": False},
            ("--no-force-min-tokens",),
            id="bool-true-default",
        ),
        param(
            {"max_idle_gap_cap_seconds": 5.0},
            ("--max-idle-gap-cap-seconds",),
            id="optional-float",
        ),
        param(
            {"trace_session_sample_ratio": 0.5},
            ("--trace-session-sample-ratio",),
            id="sample-ratio",
        ),
    ],
)  # fmt: skip
def test_resolver_reads_real_config_fields(
    overrides: dict[str, object], expected: tuple[str, ...]
) -> None:
    """Resolve against a REAL FileDataset, not a mock.

    A MagicMock auto-creates whatever attribute the resolver asks for, so a
    typo'd or renamed field would still "resolve" and this gate would silently
    never fire on a real config. Building the actual model is the only way the
    field names get checked.

    ``force_min_tokens`` defaults to True, so it also pins that the resolver
    keys on value-differs-from-default rather than presence -- the latter would
    flag every run.
    """
    from aiperf.common.enums import DatasetType
    from aiperf.config.dataset.config import FileDataset

    dataset = FileDataset(
        name="trace", type=DatasetType.FILE, path="/tmp/trace.jsonl", **overrides
    )

    class _Cfg:
        # _default_dataset() -> run.cfg.get_default_dataset(); the resolver
        # narrows THAT, not a raw datasets list.
        @staticmethod
        def get_default_dataset():
            return dataset

        @staticmethod
        def get_profiling_phases():
            return []

    class _Run:
        cfg = _Cfg()

    assert _resolve_replay_only_knobs(_Run()) == expected

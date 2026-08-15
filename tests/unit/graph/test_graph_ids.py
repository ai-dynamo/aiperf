# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two graph ID grammars, pinned in one place.

Confusing the template id with the per-recycle instance id previously
deadlocked spawn dispatch, and splitting node ids on ``_`` instead of ``:``
previously merged unrelated sessions. Both regressions are covered here.
"""

import pytest
from pytest import param

from aiperf.graph.ids import chain_key, split_node_id, template_id


@pytest.mark.parametrize(
    "node_id,expected",
    [
        param("sess_A:0", ("sess_A", 0), id="underscore-in-scope"),
        param("sess_A:12", ("sess_A", 12), id="multi-digit-turn"),
        param("phase:review", None, id="non-numeric-turn-is-native-id"),
        param("plan", None, id="bare-native-id"),
        param(":0", None, id="empty-scope-rejected"),
    ],
)  # fmt: skip
def test_split_node_id(node_id: str, expected: tuple[str, int] | None) -> None:
    assert split_node_id(node_id) == expected


@pytest.mark.parametrize(
    "node_id,expected",
    [
        param("sess_A:0", "sess_A", id="splits-on-colon-not-underscore"),
        param("sess_B:1", "sess_B", id="distinct-underscore-scopes-stay-distinct"),
        param("plan", "plan", id="unshaped-id-is-its-own-singleton-chain"),
    ],
)  # fmt: skip
def test_chain_key(node_id: str, expected: str) -> None:
    assert chain_key(node_id) == expected


def test_chain_key_does_not_merge_underscore_prefixed_sessions() -> None:
    """Regression: splitting on ``_`` collapsed sess_A and sess_B into ``sess``."""
    assert chain_key("sess_A:0") != chain_key("sess_B:1")


@pytest.mark.parametrize(
    "trace_id,expected",
    [
        param("t-1::3f2a", "t-1", id="strips-nonce"),
        param("t-1", "t-1", id="template-without-nonce-is-itself"),
        param("t-1::a::b", "t-1", id="only-first-delimiter-splits"),
    ],
)  # fmt: skip
def test_template_id(trace_id: str, expected: str) -> None:
    assert template_id(trace_id) == expected


def test_template_id_collapses_recycle_instances_of_one_template() -> None:
    """Every recycle instance must read the same build-time store."""
    assert template_id("t-1::aaa") == template_id("t-1::bbb") == "t-1"

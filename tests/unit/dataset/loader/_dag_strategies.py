# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable hypothesis strategies for property-based DAG fuzzing.

Each strategy is intentionally bounded so generated examples remain small
and the loader's validator can give honest pass/fail signal in single-digit
milliseconds. The strategies build *valid* DAG JSONL line dictionaries; the
loader is expected to round-trip them without raising.

Strategies
----------

- ``session_ids(n)``: ``n`` unique, deterministic session-id strings.
- ``message_dict()``: minimal user-role chat message dict.
- ``dag_turn(...)``: a flat DagTurn dict with optional structural keys.
- ``dag_conversation(...)``: a single DagConversation line dict.
- ``dag_dataset()``: a list of DagConversation line dicts that resolve
  internally (every spawn / fork target exists). FORK children get exactly
  one parent. Pre-session spawns are added on roots only.

All strategies guarantee the resulting dataset passes
``validate_for_orchestrator_v1`` so property tests can focus on loader
semantics rather than re-deriving validity each run.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

# -- Atoms --------------------------------------------------------------------


def session_ids(n: int) -> st.SearchStrategy[list[str]]:
    """Strategy yielding a list of ``n`` unique session-id strings."""
    assert n >= 1
    return st.just([f"s{i}" for i in range(n)])


@st.composite
def message_dict(draw: st.DrawFn) -> dict[str, Any]:
    """A minimal valid OpenAI-style user message dict."""
    content = draw(
        st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=1,
            max_size=12,
        )
    )
    return {"role": "user", "content": content}


@st.composite
def dag_turn(
    draw: st.DrawFn,
    *,
    forks: list[str] | None = None,
    spawns: list[Any] | None = None,
    allow_extras: bool = True,
) -> dict[str, Any]:
    """A flat DagTurn dict.

    ``forks`` / ``spawns`` are passed through verbatim when supplied (the
    parent strategy resolves cross-conversation references). ``allow_extras``
    toggles whether optional keys (``delay``, ``extra``) get drawn.
    """
    turn: dict[str, Any] = {
        "messages": [draw(message_dict())],
    }
    if forks:
        turn["forks"] = list(forks)
    if spawns:
        turn["spawns"] = list(spawns)
    if allow_extras:
        if draw(st.booleans()):
            turn["delay"] = float(draw(st.integers(min_value=0, max_value=200)))
        if draw(st.booleans()):
            turn["extra"] = {
                "temperature": draw(st.floats(min_value=0.0, max_value=2.0))
            }
    return turn


# -- Composite dataset strategy ----------------------------------------------


@st.composite
def dag_dataset(
    draw: st.DrawFn,
    *,
    min_convs: int = 2,
    max_convs: int = 5,
    max_turns_per_conv: int = 4,
    allow_pre_session: bool = True,
    allow_delayed_join: bool = True,
) -> list[dict[str, Any]]:
    """Generate a self-resolving list of DagConversation line dicts.

    Resolution rules baked in:

    - One conversation is the root. The remainder are leaves (no children).
    - Each non-root conversation is referenced from the root at most once.
    - FORK targets get a unique parent. SPAWN targets may be reused across
      conversations but not within a single turn (loader rejects dup ids).
    - If ``allow_delayed_join`` and the root has >=3 turns, a SPAWN may use
      a ``join_at`` strictly between (spawn_turn+1, num_turns-1).
    - If ``allow_pre_session`` the root may carry a single
      ``pre_session_spawns`` reference to one leaf.
    """
    n = draw(st.integers(min_value=min_convs, max_value=max_convs))
    sids = [f"s{i}" for i in range(n)]
    root_sid = sids[0]
    leaves = sids[1:]

    # Reserve some leaves up-front for pre-session and per-turn spawns.
    pre_session_pool = list(leaves)
    pre_choice: str | None = None
    if allow_pre_session and pre_session_pool and draw(st.booleans()):
        pre_choice = pre_session_pool.pop(0)

    # Available pool for per-turn fork/spawn references.
    available = pre_session_pool[:]
    num_turns = draw(st.integers(min_value=1, max_value=max_turns_per_conv))

    root_turns: list[dict[str, Any]] = []
    used_in_turns: set[str] = set()
    for idx in range(num_turns):
        forks: list[str] = []
        spawns: list[Any] = []
        # Decide whether to attach a branch on this turn. Last turn cannot
        # fork (would orphan the parent script per loader rule).
        is_last = idx == num_turns - 1
        if available and draw(st.booleans()):
            child = available[0]
            # FORK is only legal on the *last* turn (the loader rejects
            # FORK on a non-terminal turn with no explicit join). On
            # non-terminal turns we use SPAWN; on terminal turns we coin
            # a coin between FORK and a "terminal" background SPAWN.
            if is_last and draw(st.booleans()):
                forks = [child]
                available.pop(0)
                used_in_turns.add(child)
            else:
                # SPAWN. May be delayed if room remains and we're allowed.
                if (
                    allow_delayed_join
                    and not is_last
                    and num_turns - idx >= 3
                    and draw(st.booleans())
                ):
                    join_at = draw(
                        st.integers(min_value=idx + 2, max_value=num_turns - 1)
                    )
                    spawns = [{"children": [child], "join_at": join_at}]
                else:
                    spawns = [child]
                available.pop(0)
                used_in_turns.add(child)
        root_turns.append(draw(dag_turn(forks=forks, spawns=spawns)))

    # Build the final list. Order: root first, then every referenced leaf.
    referenced = sorted(used_in_turns | ({pre_choice} if pre_choice else set()))
    lines: list[dict[str, Any]] = []
    root_line: dict[str, Any] = {"session_id": root_sid, "turns": root_turns}
    if pre_choice is not None:
        root_line["pre_session_spawns"] = [pre_choice]
    lines.append(root_line)
    for sid in referenced:
        lines.append(
            {
                "session_id": sid,
                "turns": [draw(dag_turn(allow_extras=False))],
            }
        )
    return lines

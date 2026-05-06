# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--num-conversations autodefault for dag_jsonl input.

For DAG-shaped (forking) datasets, ``--request-count`` is a literal
wire-request cap that includes fork-spawned children, so the generic
``concurrency * MULT`` default would silently truncate the DAG mid-tree.
Instead, default ``--num-conversations`` to the *root* count (sessions
not referenced by any fork list) and refuse to default
``--request-count``.
"""

from pathlib import Path

import orjson

from aiperf.common.config.user_config import UserConfig


def _write_dag_jsonl(path: Path, root_count: int, children_per_root: int) -> None:
    """Write a dag.jsonl file with ``root_count`` roots and ``children_per_root``
    children per root (total entries = root_count * (1 + children_per_root))."""
    records: list[dict] = []
    for i in range(root_count):
        children = [f"child-{i}-{j}" for j in range(children_per_root)]
        first_turn: dict = {"messages": [{"role": "user", "content": "x"}]}
        if children:
            first_turn["forks"] = children
        records.append({"session_id": f"root-{i}", "turns": [first_turn]})
    for i in range(root_count):
        for j in range(children_per_root):
            records.append(
                {
                    "session_id": f"child-{i}-{j}",
                    "turns": [{"messages": [{"role": "user", "content": "c"}]}],
                }
            )
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _make_config(dag_file: Path, **loadgen_overrides) -> UserConfig:
    return UserConfig(
        endpoint={
            "model_names": ["test-model"],
            "url": "http://localhost:8000/v1",
        },
        input={"file": str(dag_file), "custom_dataset_type": "dag_jsonl"},
        loadgen=loadgen_overrides or {"concurrency": 1},
    )


class TestDagAutodefault:
    def test_num_conversations_defaults_to_root_count(self, tmp_path: Path) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=3, children_per_root=2)
        # 3 roots + 6 children = 9 entries; should default num-conversations to 3.

        config = _make_config(dag_file, concurrency=2)

        assert config.input.conversation.num == 3

    def test_request_count_not_defaulted_for_forking_dataset(
        self, tmp_path: Path
    ) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=2, children_per_root=1)

        config = _make_config(dag_file, concurrency=2)

        # --request-count must NOT be auto-defaulted (would truncate mid-tree).
        # --num-conversations IS defaulted to the root count.
        assert config.loadgen.request_count is None
        assert config.input.conversation.num == 2

    def test_explicit_request_count_overrides_autodefault(self, tmp_path: Path) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=3, children_per_root=2)

        config = _make_config(dag_file, concurrency=2, request_count=100)

        # User-provided --request-count is honored; --num-conversations
        # is NOT autodefaulted because the user already chose a cap.
        assert config.loadgen.request_count == 100

    def test_explicit_num_conversations_overrides_root_default(
        self, tmp_path: Path
    ) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=3, children_per_root=2)

        config = UserConfig(
            endpoint={
                "model_names": ["test-model"],
                "url": "http://localhost:8000/v1",
            },
            input={
                "file": str(dag_file),
                "custom_dataset_type": "dag_jsonl",
                "conversation": {"num": 7},
            },
            loadgen={"concurrency": 2},
        )

        assert config.input.conversation.num == 7
        assert config.loadgen.request_count is None

    def test_non_dag_dataset_uses_generic_default(self, tmp_path: Path) -> None:
        # Non-forking dataset (no custom_dataset_type or single_turn) keeps
        # the generic concurrency-based --request-count autodefault.
        config = UserConfig(
            endpoint={
                "model_names": ["test-model"],
                "url": "http://localhost:8000/v1",
            },
            loadgen={"concurrency": 4},
        )

        # request_count should be auto-defaulted; num-conversations stays None.
        assert config.loadgen.request_count is not None
        assert config.input.conversation.num is None


class TestUserConfigHelpers:
    def test_is_forking_dataset_true_for_dag_jsonl(self, tmp_path: Path) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=2, children_per_root=1)
        config = _make_config(dag_file, concurrency=1)

        assert config._is_forking_dataset() is True

    def test_is_forking_dataset_false_for_non_dag(self) -> None:
        config = UserConfig(
            endpoint={
                "model_names": ["test-model"],
                "url": "http://localhost:8000/v1",
            },
        )

        assert config._is_forking_dataset() is False

    def test_count_dag_root_entries_returns_root_only(self, tmp_path: Path) -> None:
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=4, children_per_root=3)
        # 4 roots, 12 children, 16 total entries.
        config = _make_config(dag_file, concurrency=2)

        assert config._count_dag_root_entries() == 4

    def test_count_dag_root_entries_handles_no_forks(self, tmp_path: Path) -> None:
        # Five entries, none referenced by another's forks => five roots.
        dag_file = tmp_path / "x.dag.jsonl"
        _write_dag_jsonl(dag_file, root_count=5, children_per_root=0)
        config = _make_config(dag_file, concurrency=2)

        assert config._count_dag_root_entries() == 5

    def test_count_dag_root_entries_excludes_object_form_forks(
        self, tmp_path: Path
    ) -> None:
        # Object-form forks ({"child": "x", "background": true}) reference
        # children just like bare-string forks do — root counter must walk
        # both shapes or BG-fork targets get counted as roots.
        dag_file = tmp_path / "x.dag.jsonl"
        records = [
            {
                "session_id": "P",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "t0"}],
                        "forks": [{"child": "BG", "background": True}],
                    },
                    {"messages": [{"role": "user", "content": "t1"}]},
                ],
            },
            {
                "session_id": "BG",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ]
        with open(dag_file, "wb") as f:
            for r in records:
                f.write(orjson.dumps(r))
                f.write(b"\n")
        config = _make_config(dag_file, concurrency=1)

        # Only P is a root; BG is the FORK target.
        assert config._count_dag_root_entries() == 1

    def test_count_dag_root_entries_excludes_spawn_targets(
        self, tmp_path: Path
    ) -> None:
        # SPAWN children (bare-string and DagSpawn object form) must be
        # excluded from the root count; they are dispatched by their parent
        # and would over-instantiate if treated as standalone roots.
        dag_file = tmp_path / "x.dag.jsonl"
        records = [
            {
                "session_id": "P",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "t0"}],
                        "spawns": ["B", {"children": ["C", "D"], "join_at": 2}],
                    },
                    {"messages": [{"role": "user", "content": "t1"}]},
                    {"messages": [{"role": "user", "content": "t2"}]},
                ],
            },
            {
                "session_id": "B",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "C",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "D",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ]
        with open(dag_file, "wb") as f:
            for r in records:
                f.write(orjson.dumps(r))
                f.write(b"\n")
        config = _make_config(dag_file, concurrency=1)

        # Only P is a root; B / C / D are SPAWN targets.
        assert config._count_dag_root_entries() == 1

    def test_count_dag_root_entries_excludes_pre_session_spawn_targets(
        self, tmp_path: Path
    ) -> None:
        # pre_session_spawns children get fired by the orchestrator as
        # background sub-agents of the declaring root — they are not
        # standalone roots and must not be sampled as such.
        dag_file = tmp_path / "x.dag.jsonl"
        records = [
            {
                "session_id": "ROOT",
                "turns": [{"messages": [{"role": "user", "content": "t0"}]}],
                "pre_session_spawns": ["x1", "x2", "x3"],
            },
            {
                "session_id": "x1",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "x2",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "x3",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ]
        with open(dag_file, "wb") as f:
            for r in records:
                f.write(orjson.dumps(r))
                f.write(b"\n")
        config = _make_config(dag_file, concurrency=1)

        # Only ROOT is a true root.
        assert config._count_dag_root_entries() == 1

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

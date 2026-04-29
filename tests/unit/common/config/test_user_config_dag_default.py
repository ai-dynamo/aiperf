# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dag_jsonl ``--num-conversations`` auto-default.

When the user runs aiperf against a forking dataset (today: dag_jsonl)
with no explicit ``--request-count`` or ``--num-conversations``, the
config-validator should size the run to "each root in the file once"
rather than fall through to the generic
``request_count = max(MIN, concurrency * MULT)`` default.

Why: post-640c942fb, ``--request-count`` is a literal cap on wire
requests including fork-spawned children. The generic default
(typically 10) silently truncates the DAG mid-tree on every bare run.
The fix detects forking datasets and defaults
``--num-conversations = root-count`` instead, since only roots are
sampled and each sampled root produces a full DAG fanout.

Also pins the ``_count_dag_root_entries`` helper directly: it must
return the count of conversations NOT referenced by any other
conversation's ``forks`` list, regardless of how those references
spread across turns.
"""

from __future__ import annotations

from unittest.mock import mock_open, patch

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.plugin.enums import CustomDatasetType


class TestCountDagRootEntries:
    """``_count_dag_root_entries`` must return the number of conversations
    not referenced by any other conversation's ``forks`` list — the
    sampler only samples roots, so this is what
    ``--num-conversations`` should size to.
    """

    def _make_config(self, file_content: str) -> UserConfig:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            return UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    custom_dataset_type=CustomDatasetType.DAG_JSONL,
                    file="/fake/dag.jsonl",
                ),
            )

    def _root_count(self, file_content: str) -> int:
        config = self._make_config(file_content)
        with patch("builtins.open", mock_open(read_data=file_content)):
            return config._count_dag_root_entries()

    def test_one_root_two_forks_returns_one(self):
        """Classic example DAG: 1 root + 2 forks = 1 root."""
        content = (
            '{"session_id":"root","turns":[{"messages":[],"forks":["a","b"]}]}\n'
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
            '{"session_id":"b","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 1

    def test_three_independent_roots_returns_three(self):
        """No forks declared = every conversation is a root."""
        content = (
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
            '{"session_id":"b","turns":[{"messages":[]}]}\n'
            '{"session_id":"c","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 3

    def test_multiple_root_trees_returns_correct_count(self):
        """Two separate fork trees in the same file = 2 roots."""
        content = (
            '{"session_id":"r1","turns":[{"messages":[],"forks":["c1"]}]}\n'
            '{"session_id":"c1","turns":[{"messages":[]}]}\n'
            '{"session_id":"r2","turns":[{"messages":[],"forks":["c2"]}]}\n'
            '{"session_id":"c2","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 2

    def test_forks_on_later_turns_still_referenced(self):
        """A child referenced from a later turn is still NOT a root."""
        content = (
            '{"session_id":"r","turns":[{"messages":[]},{"messages":[],"forks":["x"]}]}\n'
            '{"session_id":"x","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 1

    def test_invalid_json_lines_skipped(self):
        """A malformed line in the middle shouldn't break root counting."""
        content = (
            '{"session_id":"root","turns":[{"messages":[],"forks":["a"]}]}\n'
            "this-is-not-json\n"
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 1

    def test_empty_lines_skipped(self):
        content = (
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
            "\n"
            '{"session_id":"b","turns":[{"messages":[]}]}\n'
        )
        assert self._root_count(content) == 2

    def test_no_file_returns_zero(self):
        config = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
        )
        # No custom_dataset_type, no file → still returns 0 cleanly.
        assert config._count_dag_root_entries() == 0


class TestDagJsonlBareDefault:
    """Pin the bare-command auto-default for dag_jsonl: no explicit
    ``--request-count``, no explicit ``--num-conversations``, no
    ``--benchmark-duration`` → ``--num-conversations`` is set to the
    root count, ``--request-count`` stays unset.

    The pre-fix path used to set ``--request-count`` to
    ``max(MIN, concurrency*MULT)`` which silently truncated the DAG
    mid-tree on every bare run.
    """

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_bare_default_sizes_by_root_count(self, mock_is_file, mock_exists):
        content = (
            '{"session_id":"root","turns":[{"messages":[],"forks":["a","b"]}]}\n'
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
            '{"session_id":"b","turns":[{"messages":[]}]}\n'
        )
        with patch("builtins.open", mock_open(read_data=content)):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    custom_dataset_type=CustomDatasetType.DAG_JSONL,
                    file="/fake/dag.jsonl",
                ),
            )
        # Root count = 1 (only "root" is not referenced by any forks).
        assert config.input.conversation.num == 1
        # Should NOT fall through to the generic concurrency-based default.
        assert config.loadgen.request_count is None

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_explicit_request_count_overrides_default(self, mock_is_file, mock_exists):
        """User-supplied ``--request-count`` must be honored — no
        auto-default. The user is opting into the literal-cap
        truncation behavior with eyes open."""
        content = (
            '{"session_id":"root","turns":[{"messages":[],"forks":["a"]}]}\n'
            '{"session_id":"a","turns":[{"messages":[]}]}\n'
        )
        with patch("builtins.open", mock_open(read_data=content)):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    custom_dataset_type=CustomDatasetType.DAG_JSONL,
                    file="/fake/dag.jsonl",
                ),
                loadgen=LoadGeneratorConfig(request_count=42),
            )
        assert config.loadgen.request_count == 42
        # When user sets request_count explicitly, we don't second-guess
        # by also setting num_conversations.
        assert config.input.conversation.num is None

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_file", return_value=True)
    def test_non_forking_dataset_keeps_generic_default(self, mock_is_file, mock_exists):
        """Datasets without fork semantics keep the original
        generic default. Only dag_jsonl gets the special-case path
        today.
        """
        content = '{"text": "hello"}\n'
        with patch("builtins.open", mock_open(read_data=content)):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    custom_dataset_type=CustomDatasetType.SINGLE_TURN,
                    file="/fake/single.jsonl",
                ),
            )
        # Generic default kicks in (concurrency=1, MIN=10 → 10).
        assert config.loadgen.request_count == 10
        assert config.input.conversation.num is None


class TestCountDagRootEntriesAdversarial:
    """Defensive tests for ``_count_dag_root_entries``: the counter
    runs at config-validate time on a user-provided file path. A
    malformed or empty file must not raise — the user sees a sensible
    "0 roots" answer (which then falls through to the generic
    auto-default) instead of a stacktrace.

    Skipped as overkill: unicode/special-char ids (loader's own
    validators handle this), non-string fork ids (Pydantic rejects
    upstream), self-cycles (validator rejects), giant fanouts /
    100-root files (no different from 2-root case at the counter
    layer).
    """

    def _root_count(self, file_content: str) -> int:
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                input=InputConfig(
                    custom_dataset_type=CustomDatasetType.DAG_JSONL,
                    file="/fake/dag.jsonl",
                ),
            )
        with patch("builtins.open", mock_open(read_data=file_content)):
            return config._count_dag_root_entries()

    def test_empty_file_returns_zero(self):
        """No content at all — must not raise."""
        assert self._root_count("") == 0

    def test_only_malformed_lines_returns_zero(self):
        """Garbage in the file (not JSON at all) — counter skips bad
        lines and returns whatever it could parse. With no parseable
        lines, the answer is 0 and the auto-default falls through to
        the generic concurrency-based path.
        """
        content = "not-json\nalso not json {bad}\n"
        assert self._root_count(content) == 0

    def test_missing_session_id_skipped(self):
        """A line without ``session_id`` contributes nothing — neither
        a root nor a fork target. Other valid lines still count.
        Defensive: hand-edited files lose fields sometimes."""
        content = (
            '{"turns": [{"messages": [], "forks": ["unknown"]}]}\n'
            '{"session_id": "valid_root", "turns": [{"messages": []}]}\n'
        )
        assert self._root_count(content) == 1

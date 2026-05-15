# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""--num-conversations autodefault for dag_jsonl input (v2 converter).

For DAG-shaped (forking) datasets, ``--request-count`` is a literal
wire-request cap that includes fork-spawned children, so the generic
``concurrency * MULT`` default would silently truncate the DAG mid-tree.
Instead, the CLI->YAML converter defaults ``phase.sessions`` to the *root*
count (sessions not referenced by any fork list) and refuses to default
``phase.requests``.

The DAG-walking logic lives on ``DatasetResolver`` helpers that the
converter reuses; the file I/O happens at convert time for CLI users only.
YAML-only configs must be explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.cli_config import CLIConfig


def _make_cli(**overrides) -> CLIConfig:
    """Build a minimal CLIConfig with endpoint+model, override the rest."""
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


def _write_dag_file(tmp_path: Path, lines: list[dict]) -> Path:
    """Write a dag.jsonl file with the supplied records."""
    path = tmp_path / "dag.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


class TestDagRootCountAutodefault:
    def test_three_records_one_root_defaults_sessions_to_one(self, tmp_path):
        """1 root + 2 children -> sessions=1 (not requests=3)."""
        dag = _write_dag_file(
            tmp_path,
            [
                {"session_id": "s1", "turns": [{"forks": ["s2"]}]},
                {"session_id": "s2", "turns": [{"spawns": [{"children": ["s3"]}]}]},
                {"session_id": "s3", "turns": []},
            ],
        )
        cli = _make_cli(
            input_file=str(dag),
            custom_dataset_type="dag_jsonl",
        )

        prof = build_profiling(cli)

        assert prof.get("sessions") == 1
        # The DAG default must NOT also set requests; that would defeat
        # the whole point (request_count truncates children mid-tree).
        assert "requests" not in prof

    def test_explicit_request_count_overrides_autodefault(self, tmp_path):
        """Explicit --request-count wins; no DAG autodefault applied."""
        dag = _write_dag_file(
            tmp_path,
            [
                {"session_id": "s1", "turns": [{"forks": ["s2"]}]},
                {"session_id": "s2", "turns": []},
            ],
        )
        cli = _make_cli(
            input_file=str(dag),
            custom_dataset_type="dag_jsonl",
            request_count=42,
        )

        prof = build_profiling(cli)

        assert prof["requests"] == 42
        assert "sessions" not in prof

    def test_explicit_num_conversations_overrides_autodefault(self, tmp_path):
        """Explicit --num-conversations wins; no DAG autodefault applied."""
        dag = _write_dag_file(
            tmp_path,
            [
                {"session_id": "s1", "turns": [{"forks": ["s2"]}]},
                {"session_id": "s2", "turns": []},
            ],
        )
        cli = _make_cli(
            input_file=str(dag),
            custom_dataset_type="dag_jsonl",
            conversation_num=7,
        )

        prof = build_profiling(cli)

        assert prof["sessions"] == 7

    def test_no_input_file_no_autodefault(self):
        """Bare --custom-dataset-type without --input-file falls through.

        Without a file to count, the converter cannot derive roots; it
        falls back to the generic 10-requests default like any other
        unbounded run.
        """
        cli = _make_cli(custom_dataset_type="dag_jsonl")

        prof = build_profiling(cli)

        assert prof.get("requests") == 10
        assert "sessions" not in prof

    def test_non_forking_dataset_falls_through_to_generic_default(self, tmp_path):
        """A non-DAG dataset gets the 10-requests fallback (not roots)."""
        plain = tmp_path / "plain.jsonl"
        plain.write_text('{"prompt": "hi"}\n{"prompt": "yo"}\n')
        cli = _make_cli(
            input_file=str(plain),
            custom_dataset_type="single_turn",
        )

        prof = build_profiling(cli)

        assert prof.get("requests") == 10
        assert "sessions" not in prof


@pytest.mark.parametrize(
    "shape,expected_roots",
    [
        ([{"session_id": "r1", "turns": []}], 1),
        (
            [
                {"session_id": "r1", "turns": [{"forks": ["c1", "c2"]}]},
                {"session_id": "c1", "turns": []},
                {"session_id": "c2", "turns": []},
            ],
            1,
        ),
        (
            [
                {"session_id": "r1", "turns": []},
                {"session_id": "r2", "turns": []},
                {"session_id": "r3", "turns": [{"forks": ["c1"]}]},
                {"session_id": "c1", "turns": []},
            ],
            3,
        ),
    ],
    ids=["single_root", "one_root_two_children", "three_roots_one_child"],
)
def test_root_count_shapes(tmp_path, shape, expected_roots):
    dag = _write_dag_file(tmp_path, shape)
    cli = _make_cli(input_file=str(dag), custom_dataset_type="dag_jsonl")
    prof = build_profiling(cli)
    assert prof.get("sessions") == expected_roots

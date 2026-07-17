# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-kind gating for entry-count resolution (graph #1106 signal fix).

``--request-count`` is a recycle count on the graph/file plane, not a corpus
cap: a single trace can emit many requests. So for a **file** dataset it must
NOT back-fill ``FileDataset.entries``. An explicit ``--num-dataset-entries`` /
``--num-conversations`` still caps a file dataset (single-pass semantics), and
synthetic/public datasets keep the historical ``--request-count`` back-fill.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags._converter_dataset import _resolve_entries
from aiperf.config.flags.cli_config import CLIConfig


def _loadgen(**kwargs: object) -> dict:
    """Model-dump the load-generator fields (request_count) as explicit-set."""
    return CLIConfig(**kwargs).model_dump(exclude_unset=True)


def _file_cli(tmp_path: Path, **kwargs: object) -> CLIConfig:
    """Build a real FILE-dataset CLIConfig backed by an existing input file."""
    input_file = tmp_path / "trace.jsonl"
    input_file.write_text('{"text": "hi"}\n')
    return CLIConfig(model_names=["test-model"], input_file=str(input_file), **kwargs)


def test_request_count_does_not_backfill_file_dataset_entries(
    tmp_path: Path,
) -> None:
    """File dataset + only --request-count -> entries None (no corpus cap)."""
    cli = _file_cli(tmp_path, **_loadgen(request_count=500))
    assert cli.input_file is not None
    assert "request_count" in cli.model_fields_set
    assert _resolve_entries(cli) is None


def test_explicit_num_dataset_entries_sets_file_entries(tmp_path: Path) -> None:
    """Explicit --num-dataset-entries caps a file dataset even with --request-count."""
    cli = _file_cli(
        tmp_path,
        conversation_num_dataset_entries=50,
        **_loadgen(request_count=500),
    )
    assert _resolve_entries(cli) == 50


def test_explicit_num_conversations_sets_file_entries(tmp_path: Path) -> None:
    """Explicit --num-conversations caps a file dataset (single-pass semantics)."""
    cli = _file_cli(
        tmp_path,
        conversation_num=25,
        **_loadgen(request_count=500),
    )
    assert _resolve_entries(cli) == 25


def test_synthetic_still_backfills_from_request_count() -> None:
    """Synthetic dataset keeps the --request-count -> entries back-fill."""
    cli = CLIConfig(model_names=["test-model"], **_loadgen(request_count=500))
    assert cli.input_file is None
    assert _resolve_entries(cli) == 500


def test_public_still_backfills_from_request_count() -> None:
    """Public dataset keeps the --request-count -> entries back-fill."""
    cli = CLIConfig(
        model_names=["test-model"],
        public_dataset="sharegpt",
        **_loadgen(request_count=500),
    )
    assert cli.input_file is None
    assert _resolve_entries(cli) == 500


@pytest.mark.parametrize("request_count", [None, 500])
def test_file_dataset_no_entry_source_returns_none(
    tmp_path: Path, request_count: int | None
) -> None:
    """File dataset with no explicit entry source resolves to None regardless of request_count."""
    extra = _loadgen(request_count=request_count) if request_count is not None else {}
    cli = _file_cli(tmp_path, **extra)
    assert _resolve_entries(cli) is None

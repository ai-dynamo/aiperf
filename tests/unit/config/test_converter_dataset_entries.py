# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--request-count`` back-fills dataset entries only for synthetic/public datasets, never for a file dataset (graph #1106)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

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


def test_request_count_backfills_ordinary_file_dataset_entries(
    tmp_path: Path,
) -> None:
    """An ordinary (non-graph) file dataset sizes its entry pool from ``--request-count``.

    The skip is graph-only: a graph corpus size is fixed by the recorded trace,
    so capping ``entries`` there would truncate it. Gating on "any file dataset"
    instead silently replaced the user's count with the loader's hardcoded 100.
    """
    cli = _file_cli(tmp_path, **_loadgen(request_count=500))
    assert cli.input_file is not None
    assert "request_count" in cli.model_fields_set
    assert _resolve_entries(cli) == 500


@pytest.mark.parametrize(
    ("cap_kwargs", "expected"),
    [
        param(
            {"conversation_num_dataset_entries": 50},
            50,
            id="explicit-num-dataset-entries",
        ),
        param({"conversation_num": 25}, 25, id="explicit-num-conversations"),
    ],
)  # fmt: skip
def test_explicit_cap_sets_file_entries(
    tmp_path: Path, cap_kwargs: dict, expected: int
) -> None:
    """An explicit entry cap wins on a file dataset even alongside ``--request-count`` (single-pass semantics)."""
    cli = _file_cli(tmp_path, **cap_kwargs, **_loadgen(request_count=500))
    assert _resolve_entries(cli) == expected


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        param({}, id="synthetic"),
        param({"public_dataset": "sharegpt"}, id="public"),
    ],
)  # fmt: skip
def test_non_file_datasets_still_backfill_from_request_count(
    dataset_kwargs: dict,
) -> None:
    """Synthetic and public datasets keep the historical ``--request-count`` to entries back-fill."""
    cli = CLIConfig(
        model_names=["test-model"], **dataset_kwargs, **_loadgen(request_count=500)
    )
    assert cli.input_file is None
    assert _resolve_entries(cli) == 500


def test_file_dataset_no_entry_source_returns_none(tmp_path: Path) -> None:
    """A file dataset with no entry source at all resolves to None (the loader's default applies)."""
    cli = _file_cli(tmp_path)
    assert _resolve_entries(cli) is None


def test_request_count_does_not_backfill_graph_dataset_entries(
    tmp_path: Path,
) -> None:
    """A graph workload skips the ``--request-count`` back-fill: its corpus size is fixed by the trace."""
    cli = _file_cli(
        tmp_path, graph_format="dynamo_trace", **_loadgen(request_count=500)
    )
    assert "request_count" in cli.model_fields_set
    assert _resolve_entries(cli) is None

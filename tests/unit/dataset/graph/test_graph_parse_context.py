# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphParseContext`` carries the run's dataset-selection knobs.

:func:`resolve_graph_parse_context` threads the run's default-dataset
``entries`` cap and ``synthesis.max_context_length`` into the context every
graph adapter parses through. Resolution reads the run config alone (outside
``DatasetResolver._resolve_one``), so neither the weka-HF ``org/name`` nor the
local-graph early-return in the dataset resolver can skip it -- these tests pin
that both knobs land on the resolved context, and that an unset ``entries``
resolves to ``None`` (not a coalesced default).
"""

from __future__ import annotations

from pathlib import Path

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.workload_detect import resolve_graph_parse_context
from tests.unit.conftest import make_run_from_cli

# Any existing file makes the default dataset a FileDataset; the context
# resolver reads config only and never opens the path.
_GRAPH_FIXTURE = (
    Path(__file__).resolve().parent
    / "adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
)


def test_resolve_graph_parse_context_carries_entries_and_max_context() -> None:
    """Explicit ``entries`` + ``max_context_length`` reach the parse context."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(_GRAPH_FIXTURE),
            tokenizer_name="builtin",
            conversation_num_dataset_entries=50,
            max_context_length=131072,
        )
    )

    ctx = resolve_graph_parse_context(run)

    assert ctx.num_dataset_entries == 50
    assert ctx.max_context_length == 131072


def test_resolve_graph_parse_context_unset_entries_is_none() -> None:
    """An unset ``entries`` resolves to ``None`` (no coalesced default)."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(_GRAPH_FIXTURE),
            tokenizer_name="builtin",
        )
    )

    dataset = run.cfg.get_default_dataset()
    assert "entries" not in dataset.model_fields_set

    ctx = resolve_graph_parse_context(run)

    assert ctx.num_dataset_entries is None
    assert ctx.max_context_length is None

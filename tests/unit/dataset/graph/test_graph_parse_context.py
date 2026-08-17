# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``resolve_graph_parse_context`` carries the run's dataset-selection knobs into graph parsing."""

from __future__ import annotations

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.workload_detect import resolve_graph_parse_context
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.graph.conftest import DYNAMO_NESTED_FIXTURE

# Resolution reads the run config alone -- it runs outside
# DatasetResolver._resolve_one, so neither the weka-HF `org/name` branch nor the
# local-graph early return can skip it. Any existing file is enough to make the
# default dataset a FileDataset; the resolver never opens the path.


def test_resolve_graph_parse_context_carries_entries_and_max_context() -> None:
    """Explicit entries + max_context_length reach the parse context unchanged."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(DYNAMO_NESTED_FIXTURE),
            tokenizer_name="builtin",
            conversation_num_dataset_entries=50,
            max_context_length=131072,
        )
    )

    ctx = resolve_graph_parse_context(run)

    assert ctx.num_dataset_entries == 50
    assert ctx.max_context_length == 131072


def test_resolve_graph_parse_context_unset_entries_is_none() -> None:
    """An unset entries stays None rather than being coalesced to a default."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(DYNAMO_NESTED_FIXTURE),
            tokenizer_name="builtin",
        )
    )

    dataset = run.cfg.get_default_dataset()
    assert "entries" not in dataset.model_fields_set

    ctx = resolve_graph_parse_context(run)

    assert ctx.num_dataset_entries is None
    assert ctx.max_context_length is None

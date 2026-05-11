# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for entry-count resolution in the v1 -> v2 dataset converter.

Specifically: when the user passes both ``--num-conversations N`` and
``--request-count M`` (with M != N), the materialized synthetic dataset must
contain exactly N unique conversations. The PhaseRunner separately recycles
those N conversations to fill M total requests.
"""

from __future__ import annotations

from aiperf.config.dataset import SyntheticDataset
from aiperf.config.flags._converter_dataset import _resolve_entries, build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _user(*, num_conversations: int | None, request_count: int | None) -> CLIConfig:
    """Build a minimal CLIConfig with optional num-conversations / request-count."""
    extra_kwargs: dict = {}
    if num_conversations is not None:
        extra_kwargs["conversation_num"] = num_conversations
    loadgen_kwargs: dict = {"concurrency": 2}
    if request_count is not None:
        loadgen_kwargs["request_count"] = request_count
    return CLIConfig(
        model_names=["test-model"],
        streaming=True,
        conversation_turn_mean=3,
        prompt_input_tokens_mean=64,
        **extra_kwargs,
        **CLIConfig(**loadgen_kwargs).model_dump(exclude_unset=True),
    )


def test_num_conversations_wins_over_request_count() -> None:
    """``--num-conversations N`` must beat ``--request-count M`` when both set.

    Regression: the prior precedence resolved to ``request_count`` first, so
    ``--num-conversations 10 --request-count 20`` materialized 20
    conversations instead of 10.
    """
    user = _user(num_conversations=10, request_count=20)
    assert _resolve_entries(user) == 10


def test_num_conversations_alone_resolves_to_num() -> None:
    user = _user(num_conversations=7, request_count=None)
    assert _resolve_entries(user) == 7


def test_request_count_alone_resolves_to_request_count() -> None:
    user = _user(num_conversations=None, request_count=15)
    assert _resolve_entries(user) == 15


def test_num_dataset_entries_still_wins_over_both() -> None:
    """Explicit ``--num-dataset-entries`` continues to take top priority."""
    user = CLIConfig(
        model_names=["test-model"],
        streaming=True,
        conversation_num=10,
        conversation_num_dataset_entries=42,
        prompt_input_tokens_mean=64,
        **CLIConfig(concurrency=2, request_count=20).model_dump(exclude_unset=True),
    )
    assert _resolve_entries(user) == 42


def test_neither_set_returns_none() -> None:
    user = _user(num_conversations=None, request_count=None)
    assert _resolve_entries(user) is None


def test_full_converter_synthetic_dataset_entries_eq_num_conversations() -> None:
    """End-to-end: built ``SyntheticDataset.entries`` matches ``--num-conversations``.

    Asserts the fix flows through ``build_dataset`` and survives full
    AIPerfConfig validation - the value the SyntheticDatasetComposer reads
    (``self._num_entries = dataset.entries``) is exactly 10.
    """
    user = _user(num_conversations=10, request_count=20)

    # Direct dataset-builder path
    ds_dict = build_dataset(user)
    assert ds_dict["entries"] == 10

    # Full envelope -> AIPerfConfig validation path
    aiperf_config = convert_cli_to_aiperf(user)
    main_dataset = aiperf_config.benchmark.get_default_dataset()
    assert isinstance(main_dataset, SyntheticDataset)
    assert main_dataset.entries == 10

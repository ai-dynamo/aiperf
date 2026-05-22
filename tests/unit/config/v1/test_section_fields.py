# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the per-section frozensets to the live CLIConfig field names.

The flatten plan (Tasks 2-9) pulled every field on the seven nested config
classes up to ``CLIConfig`` directly. The resolver and converter switched
from ``user.<section>.model_fields_set`` to
``user.model_fields_set & <SECTION>_FIELDS`` to preserve the "user
explicitly set this" gating across that move. These tests guard two
invariants that switch depends on:

1. Every name in a section frozenset corresponds to a real CLIConfig field
   (no typos, no leftover names from removed nested classes).
2. The seven section frozensets are pairwise disjoint, so no flattened
   field is ambiguous about which section it came from.
"""

from __future__ import annotations

from aiperf.config.flags import CLIConfig
from aiperf.config.flags._section_fields import (
    ACCURACY_FIELDS,
    ENDPOINT_FIELDS,
    INPUT_FIELDS,
    LOADGEN_FIELDS,
    OUTPUT_FIELDS,
    SWEEPING_FIELDS,
    TOKENIZER_FIELDS,
)


def test_section_fields_partition_cli_config() -> None:
    all_section_fields = (
        ENDPOINT_FIELDS
        | INPUT_FIELDS
        | OUTPUT_FIELDS
        | TOKENIZER_FIELDS
        | LOADGEN_FIELDS
        | SWEEPING_FIELDS
        | ACCURACY_FIELDS
    )
    cli_config_fields = frozenset(CLIConfig.model_fields.keys())
    missing = all_section_fields - cli_config_fields
    assert not missing, f"section fields not on CLIConfig: {sorted(missing)}"


def test_section_fields_are_disjoint() -> None:
    sections = [
        ENDPOINT_FIELDS,
        INPUT_FIELDS,
        OUTPUT_FIELDS,
        TOKENIZER_FIELDS,
        LOADGEN_FIELDS,
        SWEEPING_FIELDS,
        ACCURACY_FIELDS,
    ]
    for i, a in enumerate(sections):
        for b in sections[i + 1 :]:
            assert a.isdisjoint(b), f"collision: {a & b}"

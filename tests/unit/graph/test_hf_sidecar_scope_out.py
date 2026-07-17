# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``_looks_like_hf_dataset_id`` classification pins.

The store build does not key on this guard (every graph workload takes
``GraphStoreBuilder._build_graph_store_streaming`` regardless of source), but
the classifier still decides whether the weka loader treats an argument as an
HF repo id (``datasets.load_dataset``) or a local file/dir. These tests pin
that classification — a known weka HF repo id returns True, and an existing
local fixture path returns False.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import _looks_like_hf_dataset_id

FIXTURES = Path(__file__).parent / "fixtures"
WEKA_MIN = FIXTURES / "weka_min.json"


def test_known_weka_hf_id_triggers_hf_guard():
    # A canonical published weka HF corpus id must be classified as HF so the
    # loader fetches it via datasets.load_dataset instead of the filesystem.
    assert _looks_like_hf_dataset_id("semianalysisai/cc-traces-weka-062126") is True


def test_local_fixture_path_does_not_trigger_hf_guard():
    # An existing local filesystem Path must NOT be classified as HF — the
    # loader reads it as a local weka trace file.
    assert _looks_like_hf_dataset_id(WEKA_MIN) is False

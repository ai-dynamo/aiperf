# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from aiperf.common.enums import DatasetType
from aiperf.config.dataset.config import FileDataset


def test_filedataset_accepts_graph_format() -> None:
    ds = FileDataset(
        name="d", type=DatasetType.FILE, path="x.gz", graph_format="dynamo_trace"
    )
    assert str(ds.graph_format) == "dynamo_trace"


def test_filedataset_graph_format_defaults_none() -> None:
    ds = FileDataset(name="d", type=DatasetType.FILE, path="x.jsonl")
    assert ds.graph_format is None

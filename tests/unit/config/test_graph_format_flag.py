# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FileDataset.graph_format`` is an optional graph-adapter override defaulting to None."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.config.dataset.config import FileDataset


def test_filedataset_accepts_graph_format() -> None:
    """An explicit adapter name is accepted and carried verbatim."""
    ds = FileDataset(
        name="d", type=DatasetType.FILE, path="x.gz", graph_format="dynamo_trace"
    )
    assert str(ds.graph_format) == "dynamo_trace"


def test_filedataset_graph_format_defaults_none() -> None:
    """Omitting the field leaves it None so detection can decide."""
    dataset = FileDataset(name="d", type=DatasetType.FILE, path="x.jsonl")
    assert dataset.graph_format is None


def test_filedataset_rejects_graph_format_and_explicit_format() -> None:
    """Graph and custom loader selectors conflict before input inspection."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        FileDataset(
            name="d",
            type=DatasetType.FILE,
            path="plain.jsonl",
            format=DatasetFormat.SINGLE_TURN,
            graph_format="dynamo_trace",
        )

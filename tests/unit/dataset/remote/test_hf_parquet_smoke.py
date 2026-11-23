# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in: resolve a real third-party parquet repo with no aiperf.json.

Deselected by default (``slow``). Run with
``uv run pytest tests/unit/dataset/remote/test_hf_parquet_smoke.py -m "slow and network"``.
"""

import importlib
from pathlib import Path

import pytest

from aiperf.common.record_files import first_record
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.service import materialize_probe, probe_ref

REF = "hf://datasets/valeriol29/mooncake-traces/mooncake"


@pytest.fixture
def hub_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Undo the unit suite's HF_HUB_OFFLINE and use a throwaway HF cache."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    constants = importlib.import_module("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.asyncio
async def test_public_mooncake_parquet_repo_resolves_and_reads(
    hub_access: None,
) -> None:
    probe = await probe_ref(RemoteRef(ref=REF))
    source = await materialize_probe(probe)
    assert probe.manifest is None and probe.data_files
    record = first_record(source.local_path)
    assert record is not None
    assert {"timestamp", "input_length", "output_length", "hash_ids"} <= set(record)

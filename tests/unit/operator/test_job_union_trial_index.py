# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archived sweep children retain their multi-trial lineage."""

from pathlib import Path

import orjson

from aiperf.operator.job_union import _archived_from_summary, _sweep_linkage


def test_sweep_linkage_and_archived_job_keep_trial_index(tmp_path: Path) -> None:
    job_dir = tmp_path / "ns" / "sweep-v03-t2"
    job_dir.mkdir(parents=True)
    (job_dir / "sweep.json").write_bytes(
        orjson.dumps(
            {
                "sweep_name": "concurrency-sweep",
                "variation_index": 3,
                "variation_label": "concurrency=256",
                "trial_index": 2,
            }
        )
    )

    linkage = _sweep_linkage(job_dir)
    archived = _archived_from_summary(
        "ns",
        "sweep-v03-t2",
        {},
        mtime_iso="2026-08-01T00:00:00Z",
        name_dir=job_dir,
    )

    assert linkage.trial_index == 2
    assert archived.sweep_name == "concurrency-sweep"
    assert archived.variation_index == 3
    assert archived.trial_index == 2

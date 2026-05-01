# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

RUN_SELECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui-v1"
    / "lib"
    / "run-selector.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def test_run_selector_rows_expose_live_and_epoch_targets() -> None:
    script = f"""
        import {{ buildRunSelectorRows }} from {RUN_SELECTOR_PATH.as_uri()!r};
        const rows = buildRunSelectorRows({{
          namespace: 'bench',
          name: 'job-a',
          epochs: [
            {{ epoch: '1000', isLatest: false, mtimeEpoch: 1001, fileCount: 3 }},
            {{ epoch: '2000', isLatest: true, mtimeEpoch: 2002, fileCount: 9 }},
          ],
          current: '1000',
          hasLive: true,
        }});
        console.log(JSON.stringify(rows.map(row => ({{
          kind: row.kind,
          selected: row.selected,
          href: row.href,
          fileCount: row.fileCount,
          isLatest: row.isLatest,
        }}))));
    """
    assert _run_node(script) == (
        '[{"kind":"live","selected":false,"href":"#/jobs/bench/job-a","fileCount":null,"isLatest":false},'
        '{"kind":"epoch","selected":false,"href":"#/jobs/bench/job-a/runs/2000","fileCount":9,"isLatest":true},'
        '{"kind":"epoch","selected":true,"href":"#/jobs/bench/job-a/runs/1000","fileCount":3,"isLatest":false}]'
    )

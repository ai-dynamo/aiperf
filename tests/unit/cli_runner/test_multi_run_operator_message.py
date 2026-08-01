# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operator-mode sweep rejection message points at documentation that exists."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.cli_runner._multi_run import _reject_in_process_sweep_under_operator

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_reject_in_process_sweep_under_operator_references_existing_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every docs/ path named in the operator rejection message resolves on disk."""
    monkeypatch.setenv("AIPERF_OPERATOR_MANAGED", "1")
    plan = Mock()
    plan.is_sweep = True
    plan.configs = [Mock(), Mock()]
    variation = Mock()
    variation.values = {"concurrency": 1}
    plan.variations = [variation]

    with pytest.raises(SystemExit) as exc_info:
        _reject_in_process_sweep_under_operator(plan)

    referenced = re.findall(r"docs/[\w./-]+\.md", str(exc_info.value))
    assert referenced, "message should point at documentation"
    for rel_path in referenced:
        assert (_REPO_ROOT / rel_path).is_file(), f"missing doc: {rel_path}"

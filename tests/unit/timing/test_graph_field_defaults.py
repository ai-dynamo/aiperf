# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph replay fields declare the same defaults on both models.

``_build_credit_phase_configs`` falls back to ``CreditPhaseConfig``'s declared
default when the run has no ``FileDataset``. ``FileDataset`` declares the same
fields; the two must agree or a synthetic run and a file-backed run left at its
defaults would pace differently.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.dataset.config import FileDataset
from aiperf.timing.config import CreditPhaseConfig, _phase_field_default


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        param("replay_speedup", None, id="replay_speedup"),
        param("open_loop_replay", True, id="open_loop_replay"),
        param("open_loop_strict", False, id="open_loop_strict"),
    ],
)  # fmt: skip
def test_graph_replay_field_defaults_agree(name: str, expected: object) -> None:
    """Both models -- and the pinned literal -- declare the same default."""
    phase_default = _phase_field_default(name)
    file_default = FileDataset.model_fields[name].get_default(call_default_factory=True)

    assert phase_default == expected
    assert file_default == expected
    assert CreditPhaseConfig.model_fields[name].get_default() == expected

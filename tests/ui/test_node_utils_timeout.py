# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run_node() must not hang forever on a script that never returns control to node."""

import subprocess

import pytest

from tests.ui.node_utils import requires_node, run_node

# A repeating timer keeps node's event loop alive forever (unlike a dangling
# top-level `await`, which node's own unsettled-await detector kills quickly),
# hanging subprocess.run() forever without a timeout.
_HANGING_SCRIPT = "setInterval(() => {}, 1_000_000);"


@requires_node
@pytest.mark.timeout(15)
def test_run_node_hanging_script_raises_timeout_error() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_node(_HANGING_SCRIPT, timeout=1)

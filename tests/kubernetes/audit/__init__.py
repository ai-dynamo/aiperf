# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def aiperf_cli() -> str:
    """Absolute path to the ``aiperf`` CLI that matches the running interpreter.

    The audit shells out to the user-facing CLI on purpose -- shipping results
    incorrectly is exactly the kind of defect it exists to catch -- but a bare
    ``"aiperf"`` resolves through PATH. A globally-installed aiperf shadows the
    checkout's venv there, and if that build predates `aiperf kube` the audit
    dies with 'Unknown command "kube"' after both sides have already run.

    Resolve next to ``sys.executable`` first so the CLI always matches the code
    under test; fall back to PATH lookup for unusual layouts.
    """
    import shutil
    import sys
    from pathlib import Path

    candidate = Path(sys.executable).parent / "aiperf"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("aiperf") or "aiperf"

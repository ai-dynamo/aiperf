# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-node-kind _execute implementations for the async-dataflow TraceExecutor.

Each module in this package registers a `_execute_<kind>` function onto the
TraceExecutor.__dict__["_execute"] singledispatchmethod via a side-effect at
import time. The TraceExecutor calls `_import_dispatch_modules()` once at
__init__ to trigger registration.

Per-kind ownership:
- llm.py   -> LlmNode (the sole dispatched kind; every live producer lowers
              to LlmNode + StaticEdge)
"""

from __future__ import annotations


def _import_dispatch_modules() -> None:
    """Import every dispatch module so its registration side-effect runs.

    Called once at module import time (see bottom of this file) and also
    invoked from `TraceExecutor.__init__` defensively. Safe to call multiple
    times; singledispatch.register is idempotent for the same (cls, func) pair.
    """
    # Imports are deferred to function-call time to avoid circular import with
    # executor.py during module load. Each module registers its
    # _execute_<kind> on import.
    # noqa: F401 — imports are for side effect.
    from aiperf.graph.dispatch import (  # noqa: F401
        llm,
    )


# Auto-trigger registration on first import of this package. Any consumer
# that imports the `dispatch` subpackage thereby gets every node-kind handler
# registered without having to remember to call _import_dispatch_modules() or
# to side-effect-import individual dispatch modules. (`aiperf.graph` alone
# does not import this package; `TraceExecutor.__init__` does, defensively.)
_import_dispatch_modules()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot package for AIPerf."""

# Default the matplotlib backend to non-interactive ``Agg`` before any
# submodule (``matplotlib_export``, ``matplotlib_uncertainty``, etc.)
# imports ``matplotlib.pyplot`` — pyplot resolves the backend at first
# import and caches it, so setting MPLBACKEND afterwards is a no-op.
# Setting it here in the package ``__init__`` runs before any submodule
# import resolves and guarantees order-independence.
#
# Why Agg: renderers return figures that callers save to disk; we never
# need a GUI window. On Windows, the default TkAgg backend requires a
# working Tcl install, and uv-managed CPython on Windows ships a Tcl
# tree that the system can't always locate, producing a misleading
# TclError that masquerades as a rendering bug.
#
# Using ``os.environ.setdefault`` (vs ``matplotlib.use("Agg")``) so
# callers who explicitly want a GUI backend (Jupyter, downstream
# libraries) can set MPLBACKEND themselves and this respects it.
import os as _os

_os.environ.setdefault("MPLBACKEND", "Agg")

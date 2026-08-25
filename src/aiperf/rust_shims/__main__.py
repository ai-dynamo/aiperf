# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in external-process adapters used by native Rust features."""

from __future__ import annotations

import importlib
import sys

_SHIM_MODULES = {
    "live-streaming": "aiperf.rust_shims.live_streaming_worker",
    "slurm-generate": "aiperf.rust_shims.slurm.generate",
}


def main(arguments: list[str] | None = None) -> int:
    """Run one named Rust support shim and return its process exit code."""
    argv = sys.argv[1:] if arguments is None else arguments
    if not argv:
        print(
            f"usage: aiperf-rust-shim <{'|'.join(_SHIM_MODULES)}> [args...]",
            file=sys.stderr,
        )
        return 2

    shim_name, *shim_arguments = argv
    module_name = _SHIM_MODULES.get(shim_name)
    if module_name is None:
        print(f"unknown Rust shim: {shim_name}", file=sys.stderr)
        return 2

    try:
        module = importlib.import_module(module_name)
        shim_main = module.main
        if not callable(shim_main):
            raise TypeError(f"Rust shim {shim_name} has no callable main")
        result = shim_main(shim_arguments)
    except Exception as error:
        print(f"Rust shim {shim_name} failed: {error}", file=sys.stderr)
        return 1

    if result is None:
        return 0
    if not isinstance(result, int):
        print(
            f"Rust shim {shim_name} returned a non-integer exit code", file=sys.stderr
        )
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())

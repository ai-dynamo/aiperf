#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify installed companion discovery and native subprocess behavior."""

from __future__ import annotations

import os
import subprocess
import sysconfig
from importlib import metadata
from pathlib import Path

import orjson

from aiperf.orchestrator.runner_installation import RunnerInstallation


def main() -> int:
    """Resolve without overrides, verify capabilities, and probe stdin failure."""
    os.environ.pop("AIPERF_RUNNER_BIN", None)
    # The absolute Python interpreter launching this script remains available;
    # an empty PATH proves discovery used wheel metadata rather than tier four.
    os.environ["PATH"] = ""
    installation = RunnerInstallation.resolve()
    scripts = Path(sysconfig.get_path("scripts")).resolve()
    if installation.binary.parent != scripts:
        raise RuntimeError(
            f"companion runner resolved to {installation.binary}, expected scripts "
            f"directory {scripts}"
        )
    if installation.capabilities.get("event") != "runner_capabilities":
        raise RuntimeError("companion runner did not return runner_capabilities")
    if installation.distribution_id is None:
        raise RuntimeError("companion runner omitted exact distribution_id")
    distribution = metadata.distribution("aiperf-runner")
    manifest_text = distribution.read_text("extra_metadata/runner-build.json")
    if manifest_text is None:
        raise RuntimeError("companion wheel omitted runner-build.json metadata")
    manifest = orjson.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise RuntimeError("companion runner-build.json must contain an object")
    if manifest.get("distribution_id") != installation.distribution_id:
        raise RuntimeError(
            "companion build manifest distribution_id disagrees with the installed binary"
        )

    malformed = subprocess.run(
        [os.fspath(installation.binary)],
        input=b"{}\n",
        capture_output=True,
        check=False,
    )
    lines = [line for line in malformed.stdout.splitlines() if line.strip()]
    if malformed.returncode != 2 or len(lines) != 1:
        raise RuntimeError(
            "companion runner bootstrap smoke expected exit 2 and one terminal line; "
            f"received exit {malformed.returncode} and {len(lines)} lines"
        )
    terminal = orjson.loads(lines[0])
    if not isinstance(terminal, dict) or terminal.get("event") != "run_terminal":
        raise RuntimeError(
            "companion runner bootstrap smoke returned an invalid terminal"
        )
    if terminal.get("success") is not False:
        raise RuntimeError("malformed companion request unexpectedly succeeded")

    print(
        orjson.dumps(
            {
                "binary": os.fspath(installation.binary),
                "distribution_id": installation.distribution_id,
                "protocol_versions": installation.capabilities.get("protocol_versions"),
                "source_revision": manifest.get("source_revision"),
                "cargo_lock_sha256": manifest.get("cargo_lock_sha256"),
                "features": manifest.get("features"),
                "bootstrap_exit": malformed.returncode,
            }
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

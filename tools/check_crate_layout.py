#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enforce the crate identity/layout policy from
``specs/repository-layout.md``.

The policy separates *package identity* (globally meaningful, keeps the
``aiperf`` namespace) from *workspace path* (local, drops the redundant
prefix). This check consults ``cargo metadata`` as the sole authority for
package identity — it never derives a package name from a directory basename —
and asserts, for every workspace member:

- the umbrella package ``aiperf`` lives at ``rust/aiperf``;
- every ``aiperf-<capability>`` package lives at ``rust/<capability>``
  (the ``aiperf-`` prefix is stripped from the directory only);
- ``loadgen-core`` (the intentional cross-product exception) lives at
  ``rust/loadgen-core``;
- any other exception is explicitly allowlisted below with its rationale.

Fails with a non-zero exit code on any mismatch, so it can gate CI and
pre-commit. Requires ``cargo`` on PATH.

Usage:
    python tools/check_crate_layout.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Packages whose workspace directory intentionally does NOT follow the
# aiperf-<capability> -> rust/<capability> rule. Each entry maps the Cargo
# package name to its required directory basename under rust/, with a reason.
#
#   loadgen-core: product-neutral shared dispatch/observation contract designed
#   for both AIPerf and AI-Dynamo Mocker; it must not carry a product prefix.
#   See §3 of the naming spec.
ALLOWLISTED_EXCEPTIONS: dict[str, str] = {
    "loadgen-core": "loadgen-core",
}


def workspace_packages(repo_root: Path) -> list[tuple[str, Path]]:
    """Return (package_name, manifest_path) for every workspace member.

    Uses ``cargo metadata --no-deps`` so package identity comes from Cargo,
    never from a directory name.
    """
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(result.stdout)
    return [(pkg["name"], Path(pkg["manifest_path"])) for pkg in meta["packages"]]


def expected_directory(package_name: str) -> str:
    """Return the required ``rust/`` directory basename for a package."""
    if package_name in ALLOWLISTED_EXCEPTIONS:
        return ALLOWLISTED_EXCEPTIONS[package_name]
    if package_name == "aiperf":
        return "aiperf"
    if package_name.startswith("aiperf-"):
        return package_name[len("aiperf-") :]
    # An un-prefixed, un-allowlisted package is itself a policy violation.
    return package_name


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    crates_root = repo_root / "rust"

    violations: list[str] = []
    for name, manifest_path in workspace_packages(repo_root):
        crate_dir = manifest_path.parent
        # Only govern members that live directly under rust/ (skip excluded
        # test-fixture packages nested deeper in the tree).
        if crate_dir.parent != crates_root:
            continue

        actual = crate_dir.name

        if (
            name != "aiperf"
            and not name.startswith("aiperf-")
            and name not in ALLOWLISTED_EXCEPTIONS
        ):
            violations.append(
                f"package '{name}' ({actual}) is neither 'aiperf', an "
                f"'aiperf-<capability>' package, nor an allowlisted exception; "
                f"add it to ALLOWLISTED_EXCEPTIONS with a rationale or rename it."
            )
            continue

        expected = expected_directory(name)
        if actual != expected:
            violations.append(
                f"package '{name}' is at rust/{actual}, expected rust/{expected}"
            )

    if violations:
        print(
            f"ERROR: {len(violations)} crate layout violation(s) "
            f"(see specs/repository-layout.md):\n"
        )
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)

    print("OK: every workspace package sits at its policy-mandated rust/ directory")


if __name__ == "__main__":
    main()

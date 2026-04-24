#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate attribution CSV from a list of runtime dpkg packages.

Reads a newline-separated list of dpkg package names, queries each package's
installed version, extracts the primary license identifier from
/usr/share/doc/<pkg>/copyright, validates against the project allow/deny
policy, and writes a CSV summary.

Handles both DEP-5 structured copyright files (License: field) and plain-text
copyright files on a best-effort basis.  Normalises legacy Debian shorthand
(e.g. "GPL-2+") to canonical SPDX identifiers before validation.

Usage:
    python3 tools/generate_dpkg_attributions.py \\
        <runtime-pkgs.txt> <output.csv> <licenses.toml>
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import tomllib

# Maps legacy Debian shorthand to canonical SPDX identifiers.
# DEP-5 pre-dates SPDX; older packages use forms like "GPL-2+" instead of
# "GPL-2.0-or-later".
_DEBIAN_TO_SPDX: dict[str, str] = {
    "gpl-2": "GPL-2.0-only",
    "gpl-2+": "GPL-2.0-or-later",
    "gpl-3": "GPL-3.0-only",
    "gpl-3+": "GPL-3.0-or-later",
    "lgpl-2": "LGPL-2.0-only",
    "lgpl-2+": "LGPL-2.0-or-later",
    "lgpl-2.1": "LGPL-2.1-only",
    "lgpl-2.1+": "LGPL-2.1-or-later",
    "lgpl-3": "LGPL-3.0-only",
    "lgpl-3+": "LGPL-3.0-or-later",
    "agpl-3": "AGPL-3.0-only",
    "agpl-3+": "AGPL-3.0-or-later",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "apache-2.0": "Apache-2.0",
    "isc": "ISC",
    "mpl-2.0": "MPL-2.0",
    "cc0-1.0": "CC0-1.0",
    "public-domain": "CC0-1.0",
}

# Splits a compound SPDX expression into individual identifiers.
_SPDX_SPLIT = re.compile(r"\bAND\b|\bOR\b|\bWITH\b|[()]")


def _normalize_spdx(raw: str) -> str:
    """Normalise a Debian license shorthand or SPDX identifier.

    Takes only the first token of the license field — DEP-5 multi-line license
    fields put the identifier on the first line followed by the license text.
    """
    token = raw.strip().split()[0] if raw.strip() else "UNKNOWN"
    return _DEBIAN_TO_SPDX.get(token.lower(), token)


def _extract_spdx_ids(expression: str) -> list[str]:
    tokens = [t.strip() for t in _SPDX_SPLIT.split(expression)]
    return [t for t in tokens if t]


def _get_version(pkg: str) -> str:
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", pkg],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def _extract_license(pkg: str) -> str:
    """Extract the primary license identifier from /usr/share/doc/<pkg>/copyright.

    Scans for the first ``License:`` field, which in DEP-5 format appears in
    the header paragraph and gives the overall package license.  Falls back to
    "UNKNOWN" if the file is absent or contains no License field.
    """
    copyright_path = Path(f"/usr/share/doc/{pkg}/copyright")
    if not copyright_path.exists():
        return "UNKNOWN"

    for line in copyright_path.read_text(errors="replace").splitlines():
        if line.startswith("License:"):
            value = line[len("License:") :].strip()
            # The identifier is the first token; subsequent tokens / continuation
            # lines are the license text body.
            return value.split()[0] if value else "UNKNOWN"

    return "UNKNOWN"


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <runtime-pkgs.txt> <output.csv> <licenses.toml>",
            file=sys.stderr,
        )
        sys.exit(1)

    pkgs_path = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    config_path = Path(sys.argv[3])

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    allow: set[str] = set(config["licenses"]["allow"])
    deny: set[str] = set(config["licenses"]["deny"])

    # dpkg exceptions keyed by (name, version) — dpkg-typed entries only
    exceptions: dict[tuple[str, str], dict] = {
        (e["name"], e["version"]): e
        for e in config.get("exceptions", [])
        if e.get("type") == "dpkg"
    }

    packages = [p for p in pkgs_path.read_text().splitlines() if p.strip()]

    rows: list[dict[str, str]] = []
    failures: list[str] = []

    for pkg in packages:
        version = _get_version(pkg)
        license_spdx = _normalize_spdx(_extract_license(pkg))

        if (pkg, version) in exceptions:
            license_spdx = exceptions[(pkg, version)]["spdx"]
        else:
            for spdx_id in _extract_spdx_ids(license_spdx):
                if spdx_id in deny:
                    failures.append(f"  DENIED   {pkg} ({version}): {spdx_id}")
                elif spdx_id not in allow:
                    failures.append(
                        f"  UNKNOWN  {pkg} ({version}): {spdx_id!r} not in allow list"
                    )

        rows.append(
            {
                "dependency_type": "dpkg",
                "name": pkg,
                "version": version,
                "spdx_license": license_spdx,
            }
        )

    if failures:
        print("License validation failed:\n" + "\n".join(failures), file=sys.stderr)
        sys.exit(1)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dependency_type", "name", "version", "spdx_license"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

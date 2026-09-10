#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch, verify, and pin the hand-vendored third-party JS/CSS assets used by
the operator UI and the API server's dashboards.

These UIs deliberately have no build step (no bundler, no ``node_modules``):
each browser-runtime dependency is a single upstream file, downloaded once,
pinned to an exact version, and served as a static asset via a browser
import map or a plain ``<script>``/``<link>`` tag. This script is the audit
trail for that: MANIFEST records exactly which upstream URL and SHA-256 each
vendored file came from, so "what third-party code does this ship and where
did it come from" is a command, not an archaeology exercise.

This intentionally does NOT use ``tools/generate_python_attributions.py`` /
``licenses.toml`` machinery -- that pipeline derives its inventory from
pip-licenses metadata, which has no concept of a hand-copied JS file. Feeding
these entries into the same third-party-attributions output is a reasonable
follow-up, but is a separate change.

Usage:
    tools/vendor_ui_deps.py --check           # offline: verify vendored files match MANIFEST
    tools/vendor_ui_deps.py --update           # re-fetch every entry, verify hash, write to dest
    tools/vendor_ui_deps.py --update NAME...   # re-fetch only the named entry/entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_JSON = REPO_ROOT / "package.json"


@dataclass(frozen=True)
class VendoredAsset:
    """One pinned third-party file vendored into a UI's static tree.

    Attributes:
        name: Short identifier, used to select entries with ``--update NAME``.
        npm_package: Name of the upstream npm package this file was pinned
            from (e.g. ``"@preact/signals"``) -- this is the key package.json
            (Dependabot bookkeeping only, see its description) must list at
            ``version``. Multiple MANIFEST entries commonly share one
            npm_package (e.g. every prismjs component, or a package's main
            export and a subpath export like ``preact``/``preact/hooks``).
        version: Upstream package version this file was pinned from.
        license: SPDX identifier of the upstream package's license.
        url: Exact upstream URL the file was downloaded from.
        dest: Path (relative to repo root) the file is written to.
        sha256: Expected hash of the final file contents (after ``rewrites``).
        rewrites: Global substring replacements applied to the downloaded
            text before hashing/writing -- used only to rewrite bare module
            specifiers (e.g. ``"preact"``) to the relative vendor path
            (e.g. ``"./preact.mjs"``) so the file resolves without a bundler.
            Order matters: longer/more-specific specifiers must precede
            shorter ones they overlap with (e.g. ``preact/hooks`` before
            ``preact``).
    """

    name: str
    npm_package: str
    version: str
    license: str
    url: str
    dest: str
    sha256: str
    rewrites: tuple[tuple[str, str], ...] = field(default_factory=tuple)


_OPERATOR_UI_VENDOR = "src/aiperf/operator/ui/vendor"
_STATIC_V2_VENDOR = "src/aiperf/api/static-v2/vendor"
_STATIC_VENDOR = "src/aiperf/api/static/vendor"

MANIFEST: tuple[VendoredAsset, ...] = (
    # --- operator/ui/vendor/ (cluster/operator dashboard) ---
    VendoredAsset(
        name="operator-preact",
        npm_package="preact",
        version="10.25.4",
        license="MIT",
        url="https://unpkg.com/preact@10.25.4/dist/preact.module.js",
        dest=f"{_OPERATOR_UI_VENDOR}/preact.mjs",
        sha256="1a1af6db5b7549506c0247211860a322db7145cd254cd7f1781daf3ece7a54ab",
    ),
    VendoredAsset(
        name="operator-preact-hooks",
        npm_package="preact",
        version="10.25.4",
        license="MIT",
        url="https://unpkg.com/preact@10.25.4/hooks/dist/hooks.module.js",
        dest=f"{_OPERATOR_UI_VENDOR}/preact-hooks.mjs",
        sha256="15fb4da18437928a672901b632951f3c7d3311088fae81c81e059a4b59fe9cc2",
        rewrites=(('"preact"', '"./preact.mjs"'),),
    ),
    VendoredAsset(
        name="operator-signals-core",
        npm_package="@preact/signals-core",
        version="1.9.0",
        license="MIT",
        url="https://unpkg.com/@preact/signals-core@1.9.0/dist/signals-core.module.js",
        dest=f"{_OPERATOR_UI_VENDOR}/signals-core.mjs",
        sha256="9e6b618e84b5b45b25bacc3749b6ccd5657307935ad7a099c25c73dbd9eccf39",
    ),
    VendoredAsset(
        name="operator-signals",
        npm_package="@preact/signals",
        version="2.2.0",
        license="MIT",
        url="https://unpkg.com/@preact/signals@2.2.0/dist/signals.module.js",
        dest=f"{_OPERATOR_UI_VENDOR}/signals.mjs",
        sha256="ebe86908d021ad38f7f3ab3268908355395bbfde053a984a2cda261b81d675a7",
        rewrites=(
            ('"preact/hooks"', '"./preact-hooks.mjs"'),
            ('"preact"', '"./preact.mjs"'),
            ('"@preact/signals-core"', '"./signals-core.mjs"'),
        ),
    ),
    VendoredAsset(
        name="operator-htm",
        npm_package="htm",
        version="3.1.1",
        license="Apache-2.0",
        url="https://unpkg.com/htm@3.1.1/dist/htm.module.js",
        dest=f"{_OPERATOR_UI_VENDOR}/htm.mjs",
        sha256="ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4",
    ),
    # htm-preact.mjs is a hand-written 7-line shim binding htm to the vendored
    # Preact's h()/render()/Component -- not an upstream file, not in MANIFEST.
    VendoredAsset(
        name="operator-js-yaml",
        npm_package="js-yaml",
        version="4.1.1",
        license="MIT",
        url="https://unpkg.com/js-yaml@4.1.1/dist/js-yaml.mjs",
        dest=f"{_OPERATOR_UI_VENDOR}/js-yaml.mjs",
        sha256="efbc45850bf15f0c8ee3434983f512be656002d7507dc292c7ade4449b5d57fa",
    ),
    VendoredAsset(
        name="operator-chartjs",
        npm_package="chart.js",
        version="4.5.1",
        license="MIT",
        url="https://unpkg.com/chart.js@4.5.1/dist/chart.umd.min.js",
        dest=f"{_OPERATOR_UI_VENDOR}/chart.umd.min.js",
        sha256="48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a",
    ),
    # --- api/static-v2/vendor/ (per-run dashboard, modular Preact rewrite) ---
    VendoredAsset(
        name="static-v2-preact",
        npm_package="preact",
        version="10.25.4",
        license="MIT",
        url="https://unpkg.com/preact@10.25.4/dist/preact.module.js",
        dest=f"{_STATIC_V2_VENDOR}/preact.mjs",
        sha256="1a1af6db5b7549506c0247211860a322db7145cd254cd7f1781daf3ece7a54ab",
    ),
    VendoredAsset(
        name="static-v2-preact-hooks",
        npm_package="preact",
        version="10.25.4",
        license="MIT",
        url="https://unpkg.com/preact@10.25.4/hooks/dist/hooks.module.js",
        dest=f"{_STATIC_V2_VENDOR}/preact-hooks.mjs",
        sha256="15fb4da18437928a672901b632951f3c7d3311088fae81c81e059a4b59fe9cc2",
        rewrites=(('"preact"', '"./preact.mjs"'),),
    ),
    VendoredAsset(
        name="static-v2-signals-core",
        npm_package="@preact/signals-core",
        version="1.9.0",
        license="MIT",
        url="https://unpkg.com/@preact/signals-core@1.9.0/dist/signals-core.module.js",
        dest=f"{_STATIC_V2_VENDOR}/signals-core.mjs",
        sha256="9e6b618e84b5b45b25bacc3749b6ccd5657307935ad7a099c25c73dbd9eccf39",
    ),
    VendoredAsset(
        name="static-v2-signals",
        npm_package="@preact/signals",
        version="2.2.0",
        license="MIT",
        url="https://unpkg.com/@preact/signals@2.2.0/dist/signals.module.js",
        dest=f"{_STATIC_V2_VENDOR}/signals.mjs",
        sha256="ebe86908d021ad38f7f3ab3268908355395bbfde053a984a2cda261b81d675a7",
        rewrites=(
            ('"preact/hooks"', '"./preact-hooks.mjs"'),
            ('"preact"', '"./preact.mjs"'),
            ('"@preact/signals-core"', '"./signals-core.mjs"'),
        ),
    ),
    VendoredAsset(
        name="static-v2-htm",
        npm_package="htm",
        version="3.1.1",
        license="Apache-2.0",
        url="https://unpkg.com/htm@3.1.1/dist/htm.module.js",
        dest=f"{_STATIC_V2_VENDOR}/htm.mjs",
        sha256="ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4",
    ),
    # htm-preact.mjs shim mirrors operator/ui's -- not an upstream file.
    VendoredAsset(
        name="static-v2-chartjs",
        npm_package="chart.js",
        version="4.5.1",
        license="MIT",
        url="https://unpkg.com/chart.js@4.5.1/dist/chart.umd.min.js",
        dest=f"{_STATIC_V2_VENDOR}/chart.umd.min.js",
        sha256="48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a",
    ),
    # --- api/static/vendor/ (legacy single-file dashboard, syntax highlighting) ---
    VendoredAsset(
        name="static-prism-core",
        npm_package="prismjs",
        version="1.30.0",
        license="MIT",
        url="https://unpkg.com/prismjs@1.30.0/prism.js",
        dest=f"{_STATIC_VENDOR}/prism-core.js",
        sha256="b801451d9b4cbf1857715a97ecae442e26c111bab6e19fee0e83dfda70cc2900",
    ),
    VendoredAsset(
        name="static-prism-theme",
        npm_package="prismjs",
        version="1.30.0",
        license="MIT",
        url="https://unpkg.com/prismjs@1.30.0/themes/prism-tomorrow.css",
        dest=f"{_STATIC_VENDOR}/prism-tomorrow.css",
        sha256="d1d928842f5912ea6a59bab4852e5c551e1041727e9a9e8dda8272ba4b3a82dd",
    ),
    VendoredAsset(
        name="static-prism-python",
        npm_package="prismjs",
        version="1.30.0",
        license="MIT",
        url="https://unpkg.com/prismjs@1.30.0/components/prism-python.js",
        dest=f"{_STATIC_VENDOR}/prism-python.js",
        sha256="fd84d8bedf516b82f1b212fc059d280f8f2ca7230bef6408bea6b5ce4e8e68f4",
    ),
    VendoredAsset(
        name="static-prism-bash",
        npm_package="prismjs",
        version="1.30.0",
        license="MIT",
        url="https://unpkg.com/prismjs@1.30.0/components/prism-bash.js",
        dest=f"{_STATIC_VENDOR}/prism-bash.js",
        sha256="6c67db1a4c86269dc754b588d0ad3a0cdb295044fd466ea6f66bbf01dec306bd",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(asset: VendoredAsset) -> bytes:
    with urllib.request.urlopen(asset.url, timeout=30) as resp:  # noqa: S310
        raw = resp.read()
    text = raw.decode("utf-8")
    for old, new in asset.rewrites:
        text = text.replace(old, new)
    return text.encode("utf-8")


def check() -> int:
    """Verify every vendored file on disk matches its pinned MANIFEST hash. Offline."""
    failures = []
    for asset in MANIFEST:
        path = REPO_ROOT / asset.dest
        if not path.exists():
            failures.append(f"{asset.name}: missing file {asset.dest}")
            continue
        actual = _sha256(path.read_bytes())
        if actual != asset.sha256:
            failures.append(
                f"{asset.name}: hash mismatch for {asset.dest}\n"
                f"    expected {asset.sha256}\n"
                f"    actual   {actual}"
            )
    if failures:
        print("Vendored UI dependency check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    failures = check_package_json()
    if failures:
        print("package.json / MANIFEST drift:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(MANIFEST)} vendored UI dependencies match MANIFEST and package.json."
    )
    return 0


def check_package_json() -> list[str]:
    """Verify package.json's dependencies exactly mirror MANIFEST's npm packages.

    package.json is Dependabot bookkeeping only (see its own description) --
    it is never ``npm install``ed -- but Dependabot can only propose version
    bumps for packages it lists, so it must track the same {package: version}
    set MANIFEST actually vendors, or a bump PR and the real pinned file
    silently drift apart.
    """
    expected: dict[str, str] = {}
    for asset in MANIFEST:
        prior = expected.get(asset.npm_package)
        if prior is not None and prior != asset.version:
            return [
                f"MANIFEST itself is inconsistent: {asset.npm_package} pinned to "
                f"both {prior} and {asset.version} across entries"
            ]
        expected[asset.npm_package] = asset.version

    if not PACKAGE_JSON.exists():
        return [f"{PACKAGE_JSON.relative_to(REPO_ROOT)} does not exist"]
    actual = json.loads(PACKAGE_JSON.read_text()).get("dependencies", {})

    failures = []
    for pkg, version in sorted(expected.items()):
        if pkg not in actual:
            failures.append(f"package.json missing dependency {pkg}@{version}")
        elif actual[pkg] != version:
            failures.append(
                f"package.json has {pkg}@{actual[pkg]}, MANIFEST pins {version}"
            )
    for pkg in sorted(set(actual) - set(expected)):
        failures.append(f"package.json lists {pkg} but no MANIFEST entry vendors it")
    return failures


def update(names: list[str] | None) -> int:
    """Re-fetch entries from their pinned upstream URL and write them to dest.

    Fails loudly (without writing) if the downloaded content doesn't match
    the MANIFEST's expected SHA-256 -- that mismatch means either the upstream
    URL was edited without updating the pin, or the upstream artifact changed
    out from under a supposedly-immutable versioned URL. Either way requires
    a human to look, not a silent overwrite.
    """
    selected = [a for a in MANIFEST if names is None or a.name in names]
    if names is not None:
        missing = set(names) - {a.name for a in selected}
        if missing:
            print(
                f"Unknown asset name(s): {', '.join(sorted(missing))}", file=sys.stderr
            )
            return 1

    failures = []
    for asset in selected:
        print(f"Fetching {asset.name} ({asset.version}) from {asset.url} ...")
        content = _fetch(asset)
        actual = _sha256(content)
        if actual != asset.sha256:
            failures.append(
                f"{asset.name}: downloaded content does not match pinned hash\n"
                f"    expected {asset.sha256}\n"
                f"    actual   {actual}\n"
                f"    If this version bump is intentional, update sha256 in MANIFEST."
            )
            continue
        dest_path = REPO_ROOT / asset.dest
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        print(f"  wrote {asset.dest} ({len(content)} bytes, sha256 verified)")

    if failures:
        print("\nUpdate FAILED for some assets:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify vendored files against MANIFEST (offline).",
    )
    group.add_argument(
        "--update",
        nargs="*",
        metavar="NAME",
        help="Re-fetch all (no args) or named entries from their pinned URL.",
    )
    args = parser.parse_args()

    if args.check:
        return check()
    return update(args.update or None)


if __name__ == "__main__":
    sys.exit(main())

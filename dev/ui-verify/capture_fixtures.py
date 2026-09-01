#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump every API response the operator UI's sweep pages consume.

Run this against a results-server -- either the in-cluster one via
``kubectl exec``, or a local ``create_app(results_dir=...)`` instance -- to
produce the fixture file ``serve.mjs`` replays.

Prefer pointing the harness at a live results-server (``BASE=...`` on
``shoot.mjs``) over replaying fixtures. Fixtures cannot reproduce a 404: the
replay server answers unmatched paths with an empty-but-valid body, so a page
requesting the WRONG URL looks identical to one requesting the right URL. A
real cross-sweep epoch bug hid behind exactly that for an entire review cycle.
Fixtures exist for offline work and for pinning a payload shape in a test.

The per-child ``/api/v1/jobs/{ns}/{child}`` calls and the per-epoch artifact
listings are not optional. They are what populate metrics and the artifacts
card; omitting them renders a page that looks structurally fine and is empty,
which is how a full set of green assertions once passed against a blank
winner card.

Usage:
    python3 capture_fixtures.py --base http://127.0.0.1:8098 \\
        --namespace <namespace> \\
        --sweep gemma-bo4 --sweep gemma-conc2 > fixtures/api.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def _get(base: str, path: str, timeout: float) -> Any:
    """Fetch one endpoint, recording transport failures rather than raising.

    A failed endpoint is recorded as ``{"__error__": ...}`` instead of aborting
    the capture, so one missing route cannot silently truncate the fixture set
    and leave later pages rendering empty for an unrelated reason.
    """
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - any transport error is data here
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def capture(
    base: str, namespace: str, sweeps: list[str], timeout: float
) -> dict[str, Any]:
    """Collect every endpoint the sweep pages touch, keyed by request path."""
    out: dict[str, Any] = {}
    for path in ("/api/v1/sweeps", "/api/v1/jobs", "/api/v1/results"):
        out[path] = _get(base, path, timeout)

    for sweep in sweeps:
        for suffix in ("", "/cells", "/children", "/epochs"):
            path = f"/api/v1/sweeps/{namespace}/{sweep}{suffix}"
            out[path] = _get(base, path, timeout)

        children = out.get(f"/api/v1/sweeps/{namespace}/{sweep}/children") or {}
        for child in children.get("children") or children.get("items") or []:
            name = child.get("name")
            if name:
                path = f"/api/v1/jobs/{namespace}/{name}"
                out[path] = _get(base, path, timeout)

        epochs = out.get(f"/api/v1/sweeps/{namespace}/{sweep}/epochs") or {}
        for entry in epochs.get("epochs") or []:
            epoch = entry.get("epoch") if isinstance(entry, dict) else entry
            if epoch is not None:
                path = f"/api/v1/sweeps/{namespace}/{sweep}/epochs/{epoch}/artifacts"
                out[path] = _get(base, path, timeout)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8098")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--sweep", action="append", dest="sweeps", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    fixtures = capture(args.base, args.namespace, args.sweeps, args.timeout)
    failed = [
        k for k, v in fixtures.items() if isinstance(v, dict) and v.get("__error__")
    ]
    print(json.dumps(fixtures))
    print(
        f"captured {len(fixtures) - len(failed)}/{len(fixtures)} endpoints",
        file=sys.stderr,
    )
    for key in failed:
        print(f"  FAILED {key}: {fixtures[key]['__error__']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

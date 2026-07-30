#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the per-ecosystem inventories into one CycloneDX SBOM for the runtime image.

The runtime image is distroless and carries no dpkg database, so nothing can
enumerate its contents after the build.  Everything that ships has to be
recorded while the build still has the metadata, from three sources:

* Python distributions -- taken verbatim from the ``cyclonedx-py`` scan of the
  shipped venv, which already carries purls and license expressions.
* Debian packages -- the packages whose files are explicitly copied into the
  runtime (``runtime-pkgs.txt``), resolved to version and license through the
  same helpers the dpkg attribution CSV uses.
* FFmpeg -- built from source, so it appears in no package database on either
  side of the build and would otherwise be invisible to every scanner.

The distroless base image is recorded as a component pinned to the reference it
was built against.  Its own contents are covered by the vendor's SBOM for that
image and are deliberately not enumerated here -- claiming otherwise would make
this document assert coverage it does not have.

Runs against the `python-licenses` build stage, which already holds every input:
the cyclonedx-py scan of the venv and the dpkg package list. Nothing in the
image build depends on this script, so it adds no production build step.

Usage:
    python3 tools/generate_container_sbom.py \\
        <python-sbom.cdx.json> <runtime-pkgs.txt> <output.cdx.json> \\
        --ffmpeg-version <version> --base-image <ref>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from generate_dpkg_attributions import (
    _DEBIAN_TO_SPDX,
    _extract_license,
    _get_version,
    _normalize_spdx,
)

SPEC_VERSION = "1.6"

# CycloneDX requires `license.id` to hold a valid SPDX identifier. The
# Debian-to-SPDX map is the set we can vouch for; anything outside it goes in
# `license.name` as free text rather than being asserted as an SPDX id it is not.
_KNOWN_SPDX_IDS = frozenset(_DEBIAN_TO_SPDX.values())

# Namespace for deriving a stable serial number from the component set, so an
# unchanged image produces an unchanged SBOM instead of a fresh random UUID.
_SERIAL_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _license_entry(raw: str) -> list[dict]:
    """Build a CycloneDX licenses array from a Debian shorthand or SPDX identifier."""
    if not raw or raw.strip().upper() == "UNKNOWN":
        return []
    spdx = _normalize_spdx(raw)
    # Compound expressions must use `expression`; bare identifiers use `id`.
    if any(op in spdx for op in (" AND ", " OR ", " WITH ")):
        return [{"expression": spdx}]
    return [{"license": {"id" if spdx in _KNOWN_SPDX_IDS else "name": spdx}}]


def _deb_components(pkgs_file: Path) -> list[dict]:
    """Build components for the Debian packages copied into the runtime image."""
    if not pkgs_file.is_file():
        return []

    components = []
    for name in sorted({line.strip() for line in pkgs_file.read_text().splitlines()}):
        if not name:
            continue
        version = _get_version(name)
        components.append(
            {
                "bom-ref": f"deb:{name}@{version}",
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:deb/debian/{name}@{version}",
                "licenses": _license_entry(_extract_license(name)),
                # Deliberately "owning one or more files": a -dev package can
                # appear here because a versioned .so symlink it owns was copied,
                # even though its headers and static libs never ship.
                "description": "Debian package owning one or more files copied into the runtime image",
            }
        )
    return components


def _ffmpeg_component(version: str) -> dict:
    """Build the component for the from-source FFmpeg build.

    FFmpeg is compiled in the builder stage rather than installed from a
    package, so no scanner on either side of the build can see it.
    """
    tarball = f"https://ffmpeg.org/releases/ffmpeg-{version}.tar.xz"
    return {
        "bom-ref": f"generic:ffmpeg@{version}",
        "type": "library",
        "name": "ffmpeg",
        "version": version,
        "purl": f"pkg:generic/ffmpeg@{version}?download_url={tarball}",
        "licenses": _license_entry("LGPL-2.1-or-later"),
        "description": (
            "Built from source with a narrow codec allowlist; see the Dockerfile "
            "configure invocation for the exact component set"
        ),
        "externalReferences": [{"type": "distribution", "url": tarball}],
    }


def _base_image_component(ref: str) -> dict:
    """Build the component for the distroless base the runtime image derives from."""
    name, _, version = ref.partition(":")
    return {
        "bom-ref": f"oci:{ref}",
        "type": "container",
        "name": name,
        "version": version or "unknown",
        "purl": f"pkg:oci/{name.rsplit('/', 1)[-1]}?repository_url={name}&tag={version}",
        "description": (
            "Distroless base image. Its own contents are covered by the vendor "
            "SBOM for that image and are not enumerated here"
        ),
    }


def _serial_number(components: list[dict]) -> str:
    """Derive a stable serial number from the component identities."""
    digest = hashlib.sha256(
        "\n".join(sorted(c.get("bom-ref", "") for c in components)).encode()
    ).hexdigest()
    return f"urn:uuid:{uuid.uuid5(_SERIAL_NAMESPACE, digest)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "python_sbom", type=Path, help="cyclonedx-py output for the venv"
    )
    parser.add_argument(
        "runtime_pkgs", type=Path, help="newline-separated dpkg package names"
    )
    parser.add_argument("output", type=Path, help="destination CycloneDX JSON document")
    parser.add_argument(
        "--ffmpeg-version",
        required=True,
        help="version of the from-source FFmpeg build",
    )
    parser.add_argument(
        "--base-image", required=True, help="runtime base image reference"
    )
    args = parser.parse_args()

    python_doc = json.loads(args.python_sbom.read_text())
    python_components = python_doc.get("components", [])

    deb_components = _deb_components(args.runtime_pkgs)
    components = [
        _base_image_component(args.base_image),
        *deb_components,
        _ffmpeg_component(args.ffmpeg_version),
        *python_components,
    ]

    document = {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": _serial_number(components),
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "component": {
                "bom-ref": "aiperf-runtime-image",
                "type": "container",
                "name": "aiperf",
                "description": "AIPerf runtime container image",
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "generate_container_sbom.py",
                        "description": "AIPerf in-tree SBOM merger",
                    }
                ]
            },
        },
        "components": components,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")

    print(f"Wrote {args.output} ({len(components)} components)")
    print("  base image: 1")
    print(f"  debian:     {len(deb_components)}")
    print("  ffmpeg:     1")
    print(f"  python:     {len(python_components)}")


if __name__ == "__main__":
    main()

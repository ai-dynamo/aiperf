#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer version computation script. Not part of the public package.

Usage:
  uv run python tools/version.py current
  uv run python tools/version.py dev [--container]
  uv run python tools/version.py nightly [--container] [--date YYYYMMDD]
  uv run python tools/version.py to-semver <pep440>
"""

from __future__ import annotations

import argparse

from commitizen.config import BaseConfig

from aiperf.versioning import AiperfVersionProvider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIPerf version computation (dev tool — not part of public package)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("current", help="Print base version from pyproject.toml")

    p_dev = sub.add_parser("dev", help="Print PEP 440 or OCI dev version from git state")
    p_dev.add_argument("--container", action="store_true", help="Emit OCI-safe semver")

    p_nightly = sub.add_parser("nightly", help="Print nightly alpha version")
    p_nightly.add_argument("--container", action="store_true", help="Emit OCI-safe semver")
    p_nightly.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help="Override date (8 digits, default: today UTC)",
    )

    p_semver = sub.add_parser("to-semver", help="Convert PEP 440 to OCI-safe semver")
    p_semver.add_argument("pep440", help="PEP 440 version string to convert")

    p_set = sub.add_parser("set", help="Write a version string into pyproject.toml (no commit, no tag)")
    p_set.add_argument("version", help="PEP 440 version string to write")

    args = parser.parse_args()
    provider = AiperfVersionProvider(BaseConfig())

    if args.command == "current":
        print(provider.get_version())
    elif args.command == "dev":
        print(provider.dev_version(container=args.container))
    elif args.command == "nightly":
        print(provider.nightly_version(container=args.container, date=args.date))
    elif args.command == "to-semver":
        print(AiperfVersionProvider.to_semver(args.pep440))
    elif args.command == "set":
        provider.set_version(args.version)
        print(f"Set version to {args.version}")


if __name__ == "__main__":
    main()

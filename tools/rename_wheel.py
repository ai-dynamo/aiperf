#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rename a wheel's distribution name without rebuilding.

Repacks the wheel with:
  - updated METADATA `Name:` field
  - renamed `*.dist-info/` directory
  - regenerated RECORD with fresh sha256/size for every file
  - new filename per PEP 491

Defaults the new name to "<current-name>-nightly". Version, tags, and
contents are left untouched.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import logging
import re
import tempfile
import zipfile
from pathlib import Path

log = logging.getLogger("rename_wheel")

WHEEL_FILENAME_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<ver>[^-]+)"
    r"(-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)


def escape_name(name: str) -> str:
    """Escape a distribution name for use in a wheel filename (PEP 491)."""
    return re.sub(r"[^\w\d.]+", "_", name, flags=re.UNICODE)


def sha256_record(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={b64}"


def patch_source_version_calls(
    root: Path,
    dist_info: Path,
    old_name: str,
    new_name: str,
) -> dict[str, int]:
    """Rewrite `version("old-name")` -> `version("new-name")` in wheel .py files.

    The wheel typically contains source files that call
    `importlib.metadata.version("<dist>")` to populate `__version__`. After a
    rename, that lookup fails because the installed distribution now uses a
    different name. This walks every .py file outside the .dist-info dir,
    swaps the literal, and returns counts for logging.
    """
    # Matches any of:
    #   version("<old>")            — direct import
    #   get_version("<old>")        — common `version as get_version` alias
    #   metadata.version("<old>")   — `from importlib import metadata`
    #   importlib.metadata.version("<old>") — fully qualified
    pattern = re.compile(
        r"""\b(?:(?:importlib\.)?metadata\.version|version|get_version)"""
        r"""\(\s*(["'])""" + re.escape(old_name) + r"""\1\s*\)"""
    )

    def _sub(match: re.Match) -> str:
        full = match.group(0)
        quote = match.group(1)
        # Preserve the caller identifier; just swap the string literal.
        prefix = full[: full.index("(")]
        return f"{prefix}({quote}{new_name}{quote})"

    files_touched = 0
    total_replacements = 0
    for py in root.rglob("*.py"):
        try:
            py.relative_to(dist_info)
            continue  # skip files inside the dist-info dir
        except ValueError:
            pass
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.debug("skipping non-utf8 .py file: %s", py)
            continue
        new_text, n = pattern.subn(_sub, text)
        if n:
            py.write_text(new_text, encoding="utf-8")
            files_touched += 1
            total_replacements += n
            log.debug("  patched %d call(s) in %s", n, py.relative_to(root))
    return {"files": files_touched, "replacements": total_replacements}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("wheel", type=Path, help="Path to the input .whl file")
    p.add_argument(
        "--new-name",
        help="New distribution name. Defaults to '<current>-nightly'.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write the renamed wheel (default: same dir as input).",
    )
    p.add_argument(
        "--remove",
        action="store_true",
        help="Delete the input wheel after a successful rename.",
    )
    p.add_argument(
        "--no-patch-source",
        action="store_true",
        help=(
            "Skip rewriting in-wheel .py files. By default, occurrences of "
            '`version("<old-name>")` are replaced with `version("<new-name>")` '
            "so importlib.metadata lookups keep working after the rename."
        ),
    )
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity. -v=DEBUG, -vv=DEBUG with per-file detail.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Explicit log level (overrides -v/-q).",
    )
    args = p.parse_args()

    if args.log_level:
        level = getattr(logging, args.log_level)
    elif args.quiet:
        level = logging.WARNING
    elif args.verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    trace_files = args.verbose >= 2

    src: Path = args.wheel
    log.debug("input wheel: %s", src)
    if not src.is_file():
        log.error("not a file: %s", src)
        return 1

    m = WHEEL_FILENAME_RE.match(src.name)
    if not m:
        log.error("not a valid wheel filename: %s", src.name)
        return 1

    version = m.group("ver")
    build = m.group("build")
    py_tag, abi_tag, plat_tag = m.group("py"), m.group("abi"), m.group("plat")
    log.debug(
        "parsed filename: version=%s build=%s py=%s abi=%s plat=%s",
        version,
        build,
        py_tag,
        abi_tag,
        plat_tag,
    )

    out_dir = args.output_dir or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log.debug("output dir: %s", out_dir)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log.debug("extracting to %s", root)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(root)
            log.debug("extracted %d entries", len(zf.namelist()))

        dist_info_dirs = [
            d for d in root.iterdir() if d.is_dir() and d.name.endswith(".dist-info")
        ]
        if len(dist_info_dirs) != 1:
            log.error(
                "expected exactly one .dist-info dir, found %d", len(dist_info_dirs)
            )
            return 1
        dist_info = dist_info_dirs[0]
        log.debug("found dist-info: %s", dist_info.name)

        metadata_path = dist_info / "METADATA"
        metadata_text = metadata_path.read_text(encoding="utf-8")
        name_match = re.search(r"^Name:[ \t]*(.+)$", metadata_text, flags=re.MULTILINE)
        if not name_match:
            log.error("METADATA has no Name field")
            return 1
        current_name = name_match.group(1).strip()
        new_name = args.new_name or f"{current_name}-nightly"
        log.info("distribution name: %s -> %s", current_name, new_name)

        new_metadata_text = re.sub(
            r"^Name:[ \t]*.+$",
            f"Name: {new_name}",
            metadata_text,
            count=1,
            flags=re.MULTILINE,
        )
        metadata_path.write_text(new_metadata_text, encoding="utf-8")
        log.debug("rewrote %s (%d bytes)", metadata_path.name, len(new_metadata_text))

        new_file_name_part = escape_name(new_name)
        new_dist_info = root / f"{new_file_name_part}-{version}.dist-info"
        if new_dist_info.exists() and new_dist_info != dist_info:
            log.error("target dist-info already exists: %s", new_dist_info.name)
            return 1
        if new_dist_info != dist_info:
            log.debug(
                "renaming dist-info: %s -> %s", dist_info.name, new_dist_info.name
            )
            dist_info.rename(new_dist_info)

        if not args.no_patch_source and current_name != new_name:
            patched = patch_source_version_calls(
                root, new_dist_info, current_name, new_name
            )
            log.info(
                "patched version() lookups: %d replacement(s) across %d file(s)",
                patched["replacements"],
                patched["files"],
            )

        record_rel = f"{new_dist_info.name}/RECORD"

        parts = [new_file_name_part, version]
        if build:
            parts.append(build)
        parts.extend([py_tag, abi_tag, plat_tag])
        new_wheel_path = out_dir / ("-".join(parts) + ".whl")
        log.debug("output wheel path: %s", new_wheel_path)

        files = sorted(f for f in root.rglob("*") if f.is_file())
        log.debug("packing %d files", len(files))

        with zipfile.ZipFile(new_wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            record_rows: list[tuple[str, str, str]] = []
            total_bytes = 0
            for f in files:
                rel = f.relative_to(root).as_posix()
                if rel == record_rel:
                    continue
                data = f.read_bytes()
                hashval = sha256_record(data)
                record_rows.append((rel, hashval, str(len(data))))
                zf.writestr(rel, data)
                total_bytes += len(data)
                if trace_files:
                    log.debug("  + %s (%d bytes, %s)", rel, len(data), hashval)

            record_rows.append((record_rel, "", ""))
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(record_rows)
            zf.writestr(record_rel, buf.getvalue())
            log.debug(
                "wrote RECORD with %d entries (%d uncompressed bytes total)",
                len(record_rows),
                total_bytes,
            )

    log.info("wrote: %s", new_wheel_path)

    if args.remove and new_wheel_path.resolve() != src.resolve():
        src.unlink()
        log.info("removed: %s", src)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
